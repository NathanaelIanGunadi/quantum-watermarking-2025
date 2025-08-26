import os
import logging
import numpy as np
from typing import Union, List, Dict, Optional, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# -------- logging --------
LOG_LEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s.%(msecs)03f %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("watermarks")

# Pin a specific backend
BACKEND_NAME = "ibm_brisbane"

from watermarks.utils_test import (
    draw,
    create_random_circuit,
    simulate_counts,
    probs_from_counts,
    tvd,
    pst_from_counts,
    two_qubit_gate_count,
    plot_tvd_phase,
    plot_bar_pairs,
    marginalize_drop_pos,
)

from qiskit_api.runtime_test import (
    register_ibm_runtime,
    get_backend,
    run_on_backend_counts,
    transpile_for_backend,
)


# ---------------- watermark: ancilla Ry + entangler ----------------
def apply_output_bias_watermark(
    qc: QuantumCircuit,
    num_ancilla: int = 1,
    thetas: Union[float, List[float]] = np.pi / 6,
    work_qubits: Union[int, List[int]] = 0,
    ancilla_positions: Union[int, List[int], None] = None,
    entangler: str = "cx",
) -> QuantumCircuit:
    N = qc.num_qubits
    W = N + num_ancilla

    if isinstance(thetas, list):
        assert len(thetas) == num_ancilla, "thetas length ≠ num_ancilla"
        theta_list = thetas
    else:
        theta_list = [thetas] * num_ancilla

    if isinstance(work_qubits, list):
        assert len(work_qubits) == num_ancilla, "work_qubits length ≠ num_ancilla"
        wq_list = work_qubits
    else:
        wq_list = [work_qubits] * num_ancilla

    if ancilla_positions is None:
        anc_pos = list(range(N, W))
    elif isinstance(ancilla_positions, list):
        anc_pos = ancilla_positions.copy()
    else:
        anc_pos = [ancilla_positions]

    slot = [None] * W
    for i, p in enumerate(anc_pos):
        if p < 0 or p >= W:
            raise ValueError(f"ancilla position {p} out of range 0..{W - 1}")
        if slot[p] is not None:
            raise ValueError(f"duplicate ancilla position {p}")
        slot[p] = ("anc", i)
    orig_ctr = 0
    for w in range(W):
        if slot[w] is None:
            slot[w] = ("orig", orig_ctr)
            orig_ctr += 1
    old_to_new = {old_q: new_w for new_w, (k, old_q) in enumerate(slot) if k == "orig"}

    wqc = QuantumCircuit(W, W)
    for instr, qargs, _ in qc.data:
        if instr.name == "measure":
            continue
        remapped = [wqc.qubits[old_to_new[q._index]] for q in qargs]
        wqc.append(instr, remapped, [])

    for i in range(num_ancilla):
        anc_wire = anc_pos[i]
        theta = theta_list[i]
        work_wire = old_to_new[wq_list[i]]

        wqc.ry(theta, anc_wire)
        if entangler == "cz":
            wqc.cz(anc_wire, work_wire)
        else:
            wqc.cx(anc_wire, work_wire)

    return wqc


# ---------------- metrics/run helpers ----------------
def transpile_and_metrics(
    qc: QuantumCircuit, backend
) -> Tuple[QuantumCircuit, int, int]:
    log.info(
        f"Transpiling circuit for backend={getattr(backend, 'name', backend)} (opt=2)"
    )
    tqc = transpile_for_backend(qc, backend, optimization_level=2)
    depth = tqc.depth()
    twoq = two_qubit_gate_count(tqc)
    log.info(f"Transpile done: depth={depth}, twoq={twoq}")
    return tqc, depth, twoq


def counts_backend_or_aer(
    qc: QuantumCircuit, shots: int, backend=None
) -> Dict[str, int]:
    """Try IBM Runtime; fallback to Aer with clear logging."""
    if backend is None:
        log.info("[RUN] Using AerSimulator (no backend provided).")
        return simulate_counts(qc, shots)
    try:
        bname = getattr(backend, "name", str(backend))
        log.info(f"[RUN] Submitting with IBM backend={bname}, shots={shots}")
        return run_on_backend_counts(qc, backend, shots)
    except Exception as e:
        log.warning(
            f"[WARN] Runtime path failed ({type(e).__name__}: {e}). Falling back to Aer."
        )
        return simulate_counts(qc, shots)


def sweep_phase_tvd(
    base_qc: QuantumCircuit,
    work_qubit: int,
    anc_pos: int,
    phases: List[float],
    baseline_backend=None,
    other_backends: Optional[List] = None,
    shots: int = 4096,
    entangler: str = "cx",
) -> Tuple[float, List[float]]:
    other_backends = other_backends or [baseline_backend]

    log.info("Computing baseline distribution…")
    base_counts = counts_backend_or_aer(base_qc, shots, baseline_backend)
    base_probs = probs_from_counts(base_counts)

    tvds: List[float] = []
    best_theta = None
    best_val = -1.0

    total = len(phases)
    log_every = max(1, total // 10)
    for idx, th in enumerate(phases, 1):
        if idx == 1 or idx % log_every == 0 or idx == total:
            log.info(f"[θ sweep] {idx}/{total}  θ={th:.4f} rad")
        wqc = apply_output_bias_watermark(
            base_qc,
            num_ancilla=1,
            thetas=th,
            work_qubits=work_qubit,
            ancilla_positions=anc_pos,
            entangler=entangler,
        )
        vals = []
        for be in other_backends:
            w_counts_full = counts_backend_or_aer(wqc, shots, be)
            w_counts = marginalize_drop_pos(
                w_counts_full, width=wqc.num_qubits, drop_pos=anc_pos
            )
            w_probs = probs_from_counts(w_counts)
            vals.append(tvd(base_probs, w_probs))
        mean_tvd = float(np.mean(vals))
        tvds.append(mean_tvd)
        if mean_tvd > best_val:
            best_val = mean_tvd
            best_theta = th
    log.info(f"[θ sweep] best θ={best_theta:.4f} rad, TVD={best_val:.4f}")
    return best_theta, tvds


def compute_ppa(
    n_qubits: int,
    ancilla_choices: int,
    entangler_choices: int = 1,
    single_qubit_gate_choices: int = 1,
    angle_steps: int = 6,
) -> float:
    poss = ancilla_choices * entangler_choices * single_qubit_gate_choices * angle_steps
    return 1.0 / poss if poss > 0 else 1.0


def evaluate_once(
    base_qc: QuantumCircuit,
    backend,
    shots: int,
    best_theta: float,
    work_qubit: int,
    anc_pos: int,
    entangler: str = "cx",
) -> Dict[str, float]:
    tb, depth_b, twoq_b = transpile_and_metrics(base_qc, backend)

    wqc = apply_output_bias_watermark(
        base_qc,
        num_ancilla=1,
        thetas=best_theta,
        work_qubits=work_qubit,
        ancilla_positions=anc_pos,
        entangler=entangler,
    )
    tw, depth_w, twoq_w = transpile_and_metrics(wqc, backend)

    log.info("Running baseline circuit…")
    cb = counts_backend_or_aer(tb, shots, backend)

    log.info("Running watermarked circuit…")
    cw_full = counts_backend_or_aer(tw, shots, backend)

    # Compare on work qubits only (drop ancilla)
    cw = marginalize_drop_pos(cw_full, width=tw.num_qubits, drop_pos=anc_pos)

    pb, pw = probs_from_counts(cb), probs_from_counts(cw)
    tvd_bw = tvd(pb, pw)

    pst_b = pst_from_counts(cb, tb.num_qubits)
    pst_w = pst_from_counts(cw, base_qc.num_qubits)

    return {
        "depth_base": depth_b,
        "depth_wm": depth_w,
        "twoq_base": twoq_b,
        "twoq_wm": twoq_w,
        "pst_base": pst_b,
        "pst_wm": pst_w,
        "tvd_base_vs_wm": tvd_bw,
    }


# ---------------- main ----------------
def main():
    # --- experiment knobs ---
    num_qubits = 3
    depth = 3
    work_qubit = 1
    anc_pos = num_qubits
    shots = 1024
    phases = [k * (np.pi / 6) for k in range(0, 12)]
    entangler = "cx"

    log.info("Building baseline circuit…")
    base_qc = create_random_circuit(num_qubits, depth)
    draw(base_qc, "Baseline circuit")

    log.info("Registering IBM Runtime and selecting backend…")
    service = register_ibm_runtime()
    baseline_backend = get_backend(service, name=BACKEND_NAME)
    log.info(f"[OK] Selected backend: {baseline_backend.name}")
    other_backends = [baseline_backend]

    best_theta, tvds = sweep_phase_tvd(
        base_qc,
        work_qubit=work_qubit,
        anc_pos=anc_pos,
        phases=phases,
        baseline_backend=baseline_backend,
        other_backends=other_backends,
        shots=shots,
        entangler=entangler,
    )
    log.info(f"Chosen phase (max TVD): θ = {best_theta:.4f} rad")
    plot_tvd_phase(phases, tvds, "TVD vs Phase (baseline vs watermarked)")

    log.info("Collecting final metrics…")
    metrics = evaluate_once(
        base_qc,
        backend=baseline_backend,
        shots=shots,
        best_theta=best_theta,
        work_qubit=work_qubit,
        anc_pos=anc_pos,
        entangler=entangler,
    )
    for k, v in metrics.items():
        log.info(f"{k}: {v}")

    labels = ["Depth", "2Q gates", "PST", "TVD(base,wm)"]
    base_vals = [metrics["depth_base"], metrics["twoq_base"], metrics["pst_base"], 0.0]
    wm_vals = [
        metrics["depth_wm"],
        metrics["twoq_wm"],
        metrics["pst_wm"],
        metrics["tvd_base_vs_wm"],
    ]
    plot_bar_pairs(
        labels,
        base_vals,
        wm_vals,
        ylabel="Value",
        legend_a="Baseline",
        legend_b="Watermarked",
        title="Post-transpile metrics (opt level 2)",
    )

    ppa = compute_ppa(
        n_qubits=num_qubits,
        ancilla_choices=num_qubits + 1,
        entangler_choices=1 if entangler == "cx" else 2,
        single_qubit_gate_choices=1,
        angle_steps=6,
    )
    log.info(f"Approximate PPA (ancilla-only model): {ppa:.3e}")


if __name__ == "__main__":
    main()
