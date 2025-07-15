# output_distribution_watermark.py

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Union, List

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator


def draw(qc: QuantumCircuit, title: str):
    """Draw the given circuit with Matplotlib."""
    fig = qc.draw(output="mpl", plot_barriers=True)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def simulate_counts(qc: QuantumCircuit, shots: int = 5000) -> Counter:
    """
    Simulate the given circuit on AerSimulator (density matrix).
    Builds a fresh circuit with N qubits & N clbits,
    copies over every gate (ignoring measurements),
    injects a full-qubit measure, then runs.
    Returns a Counter of bitstring results.
    """
    backend = AerSimulator(method="density_matrix")
    N = qc.num_qubits
    circ = QuantumCircuit(N, N)

    # Copy every non-measure instruction, remapping qubits by _index
    for instr, qargs, _ in qc.data:
        if instr.name == "measure":
            continue
        new_qargs = [circ.qubits[q._index] for q in qargs]
        circ.append(instr, new_qargs, [])

    # Measure all qubits
    circ.measure(range(N), range(N))

    t_qc = transpile(circ, backend, optimization_level=2)
    job = backend.run(t_qc, shots=shots)
    return Counter(job.result().get_counts())


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
    copying every original gate one‐by‐one into the remaining wires,
    then on each ancilla i doing:

        RY(theta_i), CX(ancilla, work_wire), CX(ancilla, work_wire)

    Inputs:
      - qc: the original N‑qubit circuit.
      - num_ancilla: how many ancilla wires to insert.
      - thetas: single angle or list of length num_ancilla.
      - work_qubits: single int or list of length num_ancilla,
        referring to original‐circuit qubit indices [0..N−1].
      - ancilla_positions: single wire _index or list of length num_ancilla
        in [0..N+num_ancilla−1] where you want those ancillas.
        If None, they default to [N, N+1, …].

    Returns:
      A brand‐new QuantumCircuit on W = N+num_ancilla qubits.
    """
    N = qc.num_qubits
    W = N + num_ancilla

    # --- normalize thetas
    if isinstance(thetas, list):
        assert len(thetas) == num_ancilla, "thetas length ≠ num_ancilla"
        theta_list = thetas
    else:
        theta_list = [thetas] * num_ancilla

    # --- normalize work_qubits
    if isinstance(work_qubits, list):
        assert len(work_qubits) == num_ancilla, "work_qubits length ≠ num_ancilla"
        wq_list = work_qubits
    else:
        wq_list = [work_qubits] * num_ancilla

    # --- normalize ancilla_positions
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

    slot = [None] * W
    for i, p in enumerate(anc_pos):
        slot[p] = ("anc", i)
    orig_ctr = 0
    for w in range(W):
        if slot[w] is None:
            slot[w] = ("orig", orig_ctr)
            orig_ctr += 1

    old_to_new = {
        old_q: new_wire for new_wire, (kind, old_q) in enumerate(slot) if kind == "orig"
    }

    wqc = QuantumCircuit(W, W)

    for instr, qargs, _ in qc.data:
        if instr.name == "measure":
            continue
        remapped = [wqc.qubits[old_to_new[q._index]] for q in qargs]
        wqc.append(instr, remapped, [])

    for i in range(num_ancilla):
        theta = theta_list[i]
        original_work = wq_list[i]
        new_work_wire = old_to_new[original_work]
        anc_wire = anc_pos[i]
        wqc.ry(theta, anc_wire)
        wqc.cx(anc_wire, new_work_wire)
        wqc.cx(anc_wire, new_work_wire)

    return wqc


def detect_output_bias(
    qc: QuantumCircuit,
    thetas: Union[float, List[float]] = np.pi / 6,
    ancilla_positions: Union[int, List[int], None] = None,
    shots: int = 5000,
    tol: float = 0.02,
):
    """
    Mirror the same ancilla_positions logic and measure bias
    before/after transpilation.
    """
    W = qc.num_qubits

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

    if isinstance(thetas, list):
        theta_list = thetas
    else:
        theta_list = [thetas] * len(anc_pos)

    before = simulate_counts(qc, shots)

    backend = AerSimulator(method="density_matrix")
    tc = transpile(qc, backend, optimization_level=2)
    draw(tc, "Transpiled Watermarked Circuit")

    after = simulate_counts(tc, shots)

    detected, b0, b1, exp = [], [], [], []
    for i, w in enumerate(anc_pos):
        expected = np.sin(theta_list[i] / 2) ** 2
        exp.append(expected)

        obs0 = sum(c for bs, c in before.items() if bs[::-1][w] == "1") / sum(
            before.values()
        )
        obs1 = sum(c for bs, c in after.items() if bs[::-1][w] == "1") / sum(
            after.values()
        )

        print(
            f"\nAncilla@wire {w}: before={obs0:.4f}, after={obs1:.4f}, expected={expected:.4f}"
        )
        b0.append(obs0)
        b1.append(obs1)
        detected.append(abs(obs1 - expected) <= tol)

    return detected, b0, b1, exp


def classical_fidelity(dist1: Counter, dist2: Counter) -> float:
    """Bhattacharyya coefficient between two classical distributions."""
    t1, t2 = sum(dist1.values()), sum(dist2.values())
    F = 0.0
    for bs, c1 in dist1.items():
        p = c1 / t1
        q = dist2.get(bs, 0) / t2
        F += np.sqrt(p * q)
    return F


def compare_accuracy(
    orig_qc: QuantumCircuit,
    wm_qc: QuantumCircuit,
    ancilla_positions: Union[int, List[int]] = None,
    shots: int = 5000,
) -> float:
    """
    Drop out the ancilla bits from the watermarked distribution
    and compute classical fidelity against the original.
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

    d0 = simulate_counts(orig_qc, shots)
    d1 = simulate_counts(wm_qc, shots)

    marg = Counter()
    for bs, c in d1.items():
        bits = list(bs)
        for p in sorted(anc_pos, reverse=True):
            del bits[-1 - p]
        marg["".join(bits)] += c

    F = classical_fidelity(d0, marg)
    print(f"\nClassical fidelity = {F:.4f}")
    return F


def generate_dummy_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    """Produce a random circuit on `num_qubits` wires."""
    return random_circuit(num_qubits, depth, max_operands=2)


def main():
    num_qubits = 5
    depth = 6
    num_ancilla = 2
    thetas = [np.pi / 6, np.pi / 4]
    work_qubits = [3, 1]
    ancilla_positions = [1, 3]

    qc = generate_dummy_circuit(num_qubits, depth)
    draw(qc, "Original Dummy Circuit")

    wqc = apply_output_bias_watermark(
        qc, num_ancilla, thetas, work_qubits, ancilla_positions
    )
    draw(wqc, "Watermarked Circuit (ancillas at 1 & 3)")

    detected, before, after, exp = detect_output_bias(
        wqc, thetas, ancilla_positions, shots=3000, tol=0.02
    )
    print(f"\nDetection per ancilla: {detected}")

    fidelity = compare_accuracy(qc, wqc, ancilla_positions, shots=3000)
    print(f"Accuracy drop = {1 - fidelity:.4f}")


if __name__ == "__main__":
    main()
