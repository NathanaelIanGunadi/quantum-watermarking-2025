import argparse
import os
import numpy as np
from collections import Counter
from typing import Union, List, Dict, Optional

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeValenciaV2

from utils import draw, create_random_circuit, simulate_counts
from qiskit_api.qiskit_runtime import (
    register_ibm_runtime,
    get_backend,
    run_on_backend_counts,
    transpile_for_backend,
)


# ----------------------------------------------------------------------
# Backend-agnostic count runner
# ----------------------------------------------------------------------
def counts_on_choice(
    qc: QuantumCircuit, shots: int, backend_name: Optional[str]
) -> Dict[str, int]:
    """
    If backend_name is None or 'aer' -> use local Aer via simulate_counts().
    Otherwise, use your Runtime helper run_on_backend_counts() on that backend name.
    """
    if backend_name is None or backend_name.lower() == "aer":
        return simulate_counts(qc, shots)

    # Try Runtime path
    service = register_ibm_runtime()
    backend = get_backend(service, name=backend_name)
    return run_on_backend_counts(qc, backend, shots)


# ----------------------------------------------------------------------
# Watermark (your original logic, with a safe index map instead of q._index)
# ----------------------------------------------------------------------
def apply_output_bias_watermark(
    qc: QuantumCircuit,
    num_ancilla: int = 1,
    thetas: Union[float, List[float]] = np.pi / 6,
    work_qubits: Union[int, List[int]] = 0,
    ancilla_positions: Union[int, List[int], None] = None,
) -> QuantumCircuit:
    """
    Rebuild the input circuit on N + num_ancilla wires,
    placing `num_ancilla` fresh ancillas exactly at `ancilla_positions`,
    copying every original gate one-by-one into the remaining wires,
    then on each ancilla i doing:

        1) RY(theta_i) on that wire
        2) CZ(ancilla, work_wire) so the ancilla remains 'in use'
    """
    N = qc.num_qubits
    W = N + num_ancilla

    # normalize thetas
    if isinstance(thetas, list):
        assert len(thetas) == num_ancilla, "thetas length ≠ num_ancilla"
        theta_list = thetas
    else:
        theta_list = [thetas] * num_ancilla

    # normalize work_qubits
    if isinstance(work_qubits, list):
        assert len(work_qubits) == num_ancilla, "work_qubits length ≠ num_ancilla"
        wq_list = work_qubits
    else:
        wq_list = [work_qubits] * num_ancilla

    # normalize ancilla_positions
    if ancilla_positions is None:
        anc_pos = list(range(N, W))
    elif isinstance(ancilla_positions, list):
        anc_pos = ancilla_positions.copy()
    else:
        anc_pos = [ancilla_positions]

    # clamp & warn
    for i, p in enumerate(anc_pos):
        if p < 0 or p >= W:
            newp = min(max(0, p), W - 1)
            print(f"Warning: ancilla_positions[{i}] = {p} clamped to {newp}")
            anc_pos[i] = newp

    # build slot table: slot[w] = ("anc", i) or ("orig", original_index)
    slot = [None] * W
    for i, p in enumerate(anc_pos):
        if slot[p] is not None:
            raise ValueError(f"Duplicate ancilla position: {p}")
        slot[p] = ("anc", i)
    orig_ctr = 0
    for w in range(W):
        if slot[w] is None:
            slot[w] = ("orig", orig_ctr)
            orig_ctr += 1

    # map old-circuit qubit -> new wire index (avoid private q._index)
    old_to_new = {
        old_q: new_wire for new_wire, (kind, old_q) in enumerate(slot) if kind == "orig"
    }
    qindex = {qb: idx for idx, qb in enumerate(qc.qubits)}

    # new circuit on W qubits (and W clbits for consistent full-measure option later)
    wqc = QuantumCircuit(W, W)

    # copy original gates
    for instr, qargs, _ in qc.data:
        if instr.name == "measure":
            continue
        remapped = [wqc.qubits[old_to_new[qindex[q]]] for q in qargs]
        wqc.append(instr, remapped, [])

    # inject watermark blocks
    for i in range(num_ancilla):
        anc_wire = anc_pos[i]
        theta = theta_list[i]
        original_work = wq_list[i]
        work_wire = old_to_new[original_work]

        wqc.ry(theta, anc_wire)
        # wqc.cz(anc_wire, work_wire)
        wqc.h(work_wire)
        wqc.cz(anc_wire, work_wire)
        wqc.h(work_wire)

    return wqc


# ----------------------------------------------------------------------
# Detection & Fidelity (your original logic, now backend-aware)
# ----------------------------------------------------------------------
def detect_output_bias(
    qc: QuantumCircuit,
    thetas: Union[float, List[float]] = np.pi / 6,
    ancilla_positions: Union[int, List[int], None] = None,
    shots: int = 5000,
    tol: float = 0.02,
    backend_name: Optional[str] = None,
):
    """
    Same as your original, but:
      - 'before' and 'after' are both executed on the chosen backend (or Aer),
      - 'after' is transpiled for that backend.
    """
    W = qc.num_qubits

    # normalize ancilla positions
    if ancilla_positions is None:
        num_anc = len(thetas) if isinstance(thetas, list) else 1
        anc_pos = list(range(W - num_anc, W))
    elif isinstance(ancilla_positions, list):
        anc_pos = ancilla_positions.copy()
    else:
        anc_pos = [ancilla_positions]

    for i, p in enumerate(anc_pos):
        if p < 0 or p >= W:
            newp = min(max(0, p), W - 1)
            print(f"Warning: detect pos {p} clamped to {newp}")
            anc_pos[i] = newp

    # normalize thetas
    theta_list = thetas if isinstance(thetas, list) else [thetas] * len(anc_pos)

    # transpile for chosen backend & draw
    if backend_name is None or backend_name.lower() == "aer":
        backend = AerSimulator()
    else:
        service = register_ibm_runtime()
        backend = get_backend(service, name=backend_name)
    tc = transpile_for_backend(qc, backend, optimization_level=2)
    draw(tc, "Transpiled Watermarked Circuit")

    # simulate post-transpile
    after = counts_on_choice(qc, shots, backend_name)

    detected, res, exp = [], [], []
    for i, w in enumerate(anc_pos):
        expected = float(np.sin(theta_list[i] / 2) ** 2)
        exp.append(expected)

        # NOTE: counts keys are big-endian; select bit w as bs[::-1][w]
        tot = sum(after.values())
        obs = sum(c for bs, c in after.items() if bs[::-1][w] == "1") / max(1, tot)

        print(f"\nAncilla@wire {w}: result={obs:.4f}, expected={expected:.4f}")
        res.append(obs)
        detected.append(abs(obs - expected) <= tol)

    return detected, res, exp


def classical_fidelity(dist1: Counter, dist2: Counter) -> float:
    """Bhattacharyya coefficient between two classical distributions."""
    t1, t2 = sum(dist1.values()), sum(dist2.values())
    if t1 == 0 or t2 == 0:
        return 0.0
    F = 0.0
    for bs, c1 in dist1.items():
        p = c1 / t1
        q = dist2.get(bs, 0) / t2
        F += np.sqrt(p * q)
    return float(F)


def compare_accuracy(
    orig_qc: QuantumCircuit,
    wm_qc: QuantumCircuit,
    ancilla_positions: Union[int, List[int]] = None,
    shots: int = 5000,
    backend_name: Optional[str] = None,
) -> float:
    """
    Drop out the ancilla bits from the watermarked distribution
    and compute classical fidelity against the original.
    Both circuits are executed on the chosen backend (or Aer).
    """
    W = wm_qc.num_qubits
    if ancilla_positions is None:
        num_anc = W - orig_qc.num_qubits
        anc_pos = list(range(W - num_anc, W))
    elif isinstance(ancilla_positions, list):
        anc_pos = ancilla_positions.copy()
    else:
        anc_pos = [ancilla_positions]
    anc_pos = [min(max(0, p), W - 1) for p in anc_pos]

    d0 = counts_on_choice(orig_qc, shots, backend_name)
    d1 = counts_on_choice(wm_qc, shots, backend_name)

    # marginalize watermarked counts by dropping ancilla bit(s)
    marg = Counter()
    for bs, c in d1.items():
        bits = list(bs)
        # remove ancilla bits (bit index counted from LSB/right ⇒ delete at -1 - p)
        for p in sorted(anc_pos, reverse=True):
            del bits[-1 - p]
        marg["".join(bits)] += c

    F = classical_fidelity(d0, marg)
    print(f"\nClassical fidelity = {F:.4f}")
    return F


# ---------------------- Paper-aligned metric helpers ----------------------


def probs_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Convert raw integer counts into a probability distribution P(x).

    Args:
        counts: dict mapping bitstring -> integer count.

    Returns:
        dict mapping bitstring -> probability in [0,1], summing to 1 (if counts>0).
    """
    tot = sum(counts.values())
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in counts.items()}


def tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Total Variation Distance (TVD) between two discrete distributions.

    Definition (paper-aligned):
        TVD(P,Q) = (1/2) * Σ_x |P(x) - Q(x)|

    Intuition:
        TVD measures how distinguishable two distributions are; higher TVD
        => stronger separability. The paper maximizes TVD over the rotation angle θ.
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def pst_from_counts(counts: Dict[str, int], width: int) -> float:
    """
    Probability of Successful Trials (PST).

    Definition (paper-aligned):
        PST = Pr[ output == 0…0 ]  (on the functional qubits)
    Implementation note:
        - For a baseline (non-watermarked) circuit with 'width' measured qubits,
          we look up counts["0"*width] / total.
        - For a watermarked circuit, first drop ancilla bits (see
          `marginalize_drop_positions`) so 'width' corresponds to functional qubits.

    Args:
        counts: dict bitstring->count
        width: number of (functional) qubits considered

    Returns:
        Fraction of shots equal to the all-zero bitstring of length 'width'.
    """
    if not counts:
        return 0.0
    zero = "0" * width
    tot = sum(counts.values())
    return counts.get(zero, 0) / tot if tot > 0 else 0.0


def marginalize_drop_positions(
    counts: Dict[str, int],
    width: int,
    drop_positions: List[int],
) -> Dict[str, int]:
    """
    Drop specific wire indices from bitstrings and aggregate counts.

    Bit-ordering (Qiskit):
        In the string 'b_{n-1} ... b_1 b_0', wire index w corresponds to
        counts_key[::-1][w]. To drop wire 'w', remove character at index -1-w.

    Args:
        counts: dict bitstring->count (all keys must have length >= width)
        width: total number of qubits represented in keys
        drop_positions: list of 0-based wire indices to remove (e.g., ancilla wires)

    Returns:
        dict new_bitstring->count over the remaining (functional) wires.
    """
    if not drop_positions:
        return dict(counts)

    drop_sorted = sorted(set(drop_positions), reverse=True)
    out: Dict[str, int] = {}
    for bs, c in counts.items():
        bits = list(bs)
        for p in drop_sorted:
            # delete the character corresponding to wire p
            del bits[-1 - p]
        key = "".join(bits)
        out[key] = out.get(key, 0) + int(c)
    return out


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    """
    Count 2-qubit (or larger) gates in a circuit.

    Paper reports post-transpile 2Q gate overhead; this matches that usage.
    """
    n = 0
    for inst, qargs, _ in qc.data:
        if len(qargs) >= 2:
            n += 1
    return n


def _resolve_backend(backend_name: Optional[str]):
    """
    Return a backend object for transpilation:
      - 'aer' or None -> AerSimulator()
      - otherwise -> IBM backend via your Runtime helpers.
    """
    if backend_name is None or backend_name.lower() == "aer":
        return AerSimulator()
    service = register_ibm_runtime()
    return get_backend(service, name=backend_name)


def compute_paper_metrics(
    orig_qc: QuantumCircuit,
    wm_qc: QuantumCircuit,
    *,
    ancilla_positions: Optional[List[int]] = None,
    backend_name: Optional[str] = None,
    shots: int = 4096,
    opt_level: int = 2,
) -> Dict[str, float]:
    """
    Compute the paper’s headline metrics on a chosen backend:

    - depth_base, depth_wm: post-transpile circuit depths
    - twoq_base, twoq_wm: post-transpile 2-qubit gate counts
    - pst_base: PST of baseline (Pr[0…0] on functional wires)
    - pst_wm: PST of watermarked circuit AFTER dropping ancilla wires
    - tvd_base_vs_wm: TVD between baseline probs and WATERMARKED probs
                      on functional wires (ancilla dropped)

    This mirrors Roy & Ghosh (2024): compare functional outputs only,
    compute overheads after transpilation, and use TVD to quantify
    separability between baseline and watermarked distributions.
    """
    # --- transpile for the chosen backend
    backend = _resolve_backend(backend_name)
    tb = transpile_for_backend(orig_qc, backend, optimization_level=opt_level)
    tw = transpile_for_backend(wm_qc, backend, optimization_level=opt_level)

    depth_base = tb.depth()
    depth_wm = tw.depth()
    twoq_base = two_qubit_gate_count(tb)
    twoq_wm = two_qubit_gate_count(tw)

    # --- execute and collect counts
    cb = counts_on_choice(tb, shots, backend_name)
    cw_full = counts_on_choice(tw, shots, backend_name)

    # --- drop ancilla (so we compare only functional wires)
    W = tw.num_qubits
    if ancilla_positions is None:
        num_anc = W - orig_qc.num_qubits
        anc_pos = list(range(W - num_anc, W))
    elif isinstance(ancilla_positions, list):
        anc_pos = ancilla_positions.copy()
    else:
        anc_pos = [ancilla_positions]
    anc_pos = [min(max(0, p), W - 1) for p in anc_pos]

    cw = marginalize_drop_positions(cw_full, width=W, drop_positions=anc_pos)

    # --- distributions & metrics
    pb = probs_from_counts(cb)
    pw = probs_from_counts(cw)
    tvd_bw = tvd(pb, pw)

    pst_b = pst_from_counts(cb, width=tb.num_qubits)
    pst_w = pst_from_counts(cw, width=orig_qc.num_qubits)

    return dict(
        depth_base=depth_base,
        depth_wm=depth_wm,
        twoq_base=twoq_base,
        twoq_wm=twoq_wm,
        pst_base=pst_b,
        pst_wm=pst_w,
        tvd_base_vs_wm=tvd_bw,
    )


def sweep_theta_tvd(
    base_qc: QuantumCircuit,
    *,
    thetas: List[float],
    num_ancilla: int = 1,
    work_qubits: Union[int, List[int]] = 0,
    ancilla_positions: Union[int, List[int], None] = None,
    backend_name: Optional[str] = None,
    shots: int = 4096,
    opt_level: int = 2,
) -> Dict[str, Union[float, List[float]]]:
    """
    Sweep rotation angle(s) θ and compute TVD vs. the non-watermarked baseline.
    This reproduces the paper’s “pick θ that maximizes TVD” procedure.

    Returns:
        {
          "best_theta": float,
          "tvds": List[float]  # TVD for each theta in order,
          "baseline_probs": Dict[str,float],
        }
    """
    # baseline once
    backend = _resolve_backend(backend_name)
    tb = transpile_for_backend(base_qc, backend, optimization_level=opt_level)
    cb = counts_on_choice(tb, shots, backend_name)
    pb = probs_from_counts(cb)

    tvds: List[float] = []
    best_theta = None
    best_val = -1.0

    # normalize helpers for anc_pos/work_qubits
    N = base_qc.num_qubits
    if isinstance(work_qubits, list):
        wq_list = work_qubits
    else:
        wq_list = [work_qubits] * num_ancilla

    if ancilla_positions is None:
        anc_pos = list(range(N, N + num_ancilla))
    elif isinstance(ancilla_positions, list):
        anc_pos = ancilla_positions.copy()
    else:
        anc_pos = [ancilla_positions]

    for theta in thetas:
        wqc = apply_output_bias_watermark(
            base_qc,
            num_ancilla=num_ancilla,
            thetas=theta if num_ancilla == 1 else [theta] * num_ancilla,
            work_qubits=wq_list,
            ancilla_positions=anc_pos,
        )
        tw = transpile_for_backend(wqc, backend, optimization_level=opt_level)
        cw_full = counts_on_choice(tw, shots, backend_name)
        cw = marginalize_drop_positions(
            cw_full, width=tw.num_qubits, drop_positions=anc_pos
        )
        pw = probs_from_counts(cw)
        val = tvd(pb, pw)
        tvds.append(val)
        if val > best_val:
            best_val = val
            best_theta = theta

    return {"best_theta": float(best_theta), "tvds": tvds, "baseline_probs": pb}


# ----------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Output-distribution watermarking (backend-aware)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("IBM_BACKEND", "aer"),
        help="Backend name (e.g., 'aer' or 'ibm_qasm_simulator' or a real device name).",
    )
    parser.add_argument(
        "--nq", type=int, default=3, help="Number of functional qubits."
    )
    parser.add_argument("--depth", type=int, default=3, help="Random circuit depth.")
    parser.add_argument(
        "--shots", type=int, default=3000, help="Number of shots for sampling."
    )
    parser.add_argument(
        "--theta-sweep",
        action="store_true",
        help="Also sweep θ in {k·π/6} to report the best TVD angle.",
    )
    args = parser.parse_args()

    num_qubits = args.nq
    depth = args.depth
    shots = args.shots
    backend_name = args.backend

    print(
        f"\n[Config] backend={backend_name} | nq={num_qubits} | depth={depth} | shots={shots}"
    )

    # --- Example watermark config (same as your original) ---
    num_ancilla = 2
    thetas = [np.pi / 6, np.pi / 4]  # one θ per ancilla
    work_qubits = [2, 1]  # refer to original-circuit indices (0..N-1)
    ancilla_positions = [1, 2]  # wires in the W-qubit circuit

    # --- Build & draw original ---
    qc = create_random_circuit(num_qubits, depth)
    draw(qc, "Original Dummy Circuit")

    # --- Build & draw watermarked (with thetas above) ---
    wqc = apply_output_bias_watermark(
        qc,
        num_ancilla=num_ancilla,
        thetas=thetas,
        work_qubits=work_qubits,
        ancilla_positions=ancilla_positions,
    )
    draw(wqc, "Watermarked Circuit (ancillas at 1 & 2)")

    # --- Detect watermark bias (before/after transpile on chosen backend) ---
    detected, res, exp = detect_output_bias(
        wqc,
        thetas=thetas,
        ancilla_positions=ancilla_positions,
        shots=shots,
        tol=0.02,
        backend_name=backend_name,
    )
    print(f"\n[Detection] per-ancilla match to expectation: {detected}")

    # --- Accuracy via classical fidelity (functional wires only) ---
    fidelity = compare_accuracy(
        qc,
        wqc,
        ancilla_positions=ancilla_positions,
        shots=shots,
        backend_name=backend_name,
    )
    print(
        f"[Accuracy] Classical fidelity (orig vs wm, ancilla dropped): {fidelity:.4f}"
    )
    print(f"[Accuracy] Accuracy drop = {1 - fidelity:.4f}")

    # --- Paper-style metrics at the provided θs ---
    metrics = compute_paper_metrics(
        qc,
        wqc,
        ancilla_positions=ancilla_positions,
        backend_name=backend_name,
        shots=shots,
        opt_level=2,
    )
    depth_ov = (
        100.0
        * (metrics["depth_wm"] - metrics["depth_base"])
        / max(1, metrics["depth_base"])
    )
    twoq_ov = (
        100.0
        * (metrics["twoq_wm"] - metrics["twoq_base"])
        / max(1, metrics["twoq_base"])
    )
    pst_delta = metrics["pst_wm"] - metrics["pst_base"]

    print("\n[Paper metrics @ provided θ]")
    print(f"  depth_base        : {metrics['depth_base']}")
    print(f"  depth_wm          : {metrics['depth_wm']}  (overhead {depth_ov:+.2f}%)")
    print(f"  twoq_base         : {metrics['twoq_base']}")
    print(f"  twoq_wm           : {metrics['twoq_wm']}   (overhead {twoq_ov:+.2f}%)")
    print(f"  pst_base          : {metrics['pst_base']:.4f}")
    print(f"  pst_wm            : {metrics['pst_wm']:.4f}  (Δ {pst_delta:+.4f})")
    print(f"  tvd_base_vs_wm    : {metrics['tvd_base_vs_wm']:.4f}")

    # --- Optional: sweep θ in multiples of π/6 and report best TVD, per paper ---
    if args.theta_sweep:
        theta_grid = [k * (np.pi / 6) for k in range(0, 12)]  # {0, π/6, …, 11π/6}
        sweep = sweep_theta_tvd(
            qc,
            thetas=theta_grid,
            num_ancilla=num_ancilla,
            work_qubits=work_qubits,
            ancilla_positions=ancilla_positions,
            backend_name=backend_name,
            shots=shots,
            opt_level=2,
        )
        best_theta = sweep["best_theta"]
        print(f"\n[θ-sweep] Best θ by TVD: {best_theta:.4f} rad")

        # Rebuild watermark with best θ replicated over all ancillas
        wqc_star = apply_output_bias_watermark(
            qc,
            num_ancilla=num_ancilla,
            thetas=[best_theta] * num_ancilla,
            work_qubits=work_qubits,
            ancilla_positions=ancilla_positions,
        )
        # Metrics at θ*
        m_star = compute_paper_metrics(
            qc,
            wqc_star,
            ancilla_positions=ancilla_positions,
            backend_name=backend_name,
            shots=shots,
            opt_level=2,
        )
        d_ov_star = (
            100.0
            * (m_star["depth_wm"] - m_star["depth_base"])
            / max(1, m_star["depth_base"])
        )
        q_ov_star = (
            100.0
            * (m_star["twoq_wm"] - m_star["twoq_base"])
            / max(1, m_star["twoq_base"])
        )
        pst_delta_star = m_star["pst_wm"] - m_star["pst_base"]

        print("[Paper metrics @ θ* (max TVD)]")
        print(f"  depth_base        : {m_star['depth_base']}")
        print(
            f"  depth_wm          : {m_star['depth_wm']}  (overhead {d_ov_star:+.2f}%)"
        )
        print(f"  twoq_base         : {m_star['twoq_base']}")
        print(
            f"  twoq_wm           : {m_star['twoq_wm']}   (overhead {q_ov_star:+.2f}%)"
        )
        print(f"  pst_base          : {m_star['pst_base']:.4f}")
        print(
            f"  pst_wm            : {m_star['pst_wm']:.4f}  (Δ {pst_delta_star:+.4f})"
        )
        print(f"  tvd_base_vs_wm    : {m_star['tvd_base_vs_wm']:.4f}")

        # Compare TVD improvement relative to the provided θ set
        delta_tvd = m_star["tvd_base_vs_wm"] - metrics["tvd_base_vs_wm"]
        print(f"[θ-sweep] TVD improvement vs provided θ: {delta_tvd:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
