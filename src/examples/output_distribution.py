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
    Rebuilds a fresh circuit with equal qubits and classical bits,
    copies over quantum operations, injects measurement, then runs.
    Returns a Counter of bitstring results.
    """
    backend = AerSimulator(method="density_matrix")
    N = qc.num_qubits

    circ = QuantumCircuit(N, N)

    for instr, qargs, _ in qc.data:
        if instr.name != "measure":
            circ.append(instr, qargs, [])

    circ.measure(range(N), range(N))

    t_qc = transpile(circ, backend, optimization_level=2)
    job = backend.run(t_qc, shots=shots)
    result = job.result()
    return Counter(result.get_counts())


def apply_output_bias_watermark(
    qc: QuantumCircuit,
    num_ancilla: int = 1,
    thetas: Union[float, List[float]] = np.pi / 6,
    work_qubits: Union[int, List[int]] = 0,
) -> QuantumCircuit:
    """
    Embed one or more output-bias watermarks into any input circuit by:
      1) appending `num_ancilla` fresh ancilla qubits at indices N..N+num_ancilla-1
      2) copying the original circuit onto qubits [0..N-1]
      3) for each ancilla i:
         - apply RY(theta_i) on qubit N+i
         - apply CX(N+i, work_qubit_i) twice so the ancilla remains 'in use'

    Inputs:
      - qc: the original QuantumCircuit
      - num_ancilla: how many ancilla qubits to add
      - thetas: single float or list of floats of length `num_ancilla`
      - work_qubits: single int or list of ints of length `num_ancilla`

    Returns:
      A new QuantumCircuit with N+num_ancilla qubits and
      original_cl+num_ancilla classical bits,
      original ops on [0..N-1], plus the watermark on qubits [N..].
    """
    N = qc.num_qubits
    original_cl = qc.num_clbits

    if isinstance(thetas, list):
        assert len(thetas) == num_ancilla, (
            "Length of `thetas` must equal `num_ancilla`."
        )
        theta_list = thetas
    else:
        theta_list = [thetas] * num_ancilla

    if isinstance(work_qubits, list):
        assert len(work_qubits) == num_ancilla, (
            "Length of `work_qubits` must equal `num_ancilla`."
        )
        wq_list = work_qubits
    else:
        wq_list = [work_qubits] * num_ancilla

    wqc = QuantumCircuit(N + num_ancilla, original_cl + num_ancilla)

    orig_inst = qc.to_instruction()
    wqc.append(orig_inst, list(range(N)), list(range(original_cl)))

    for i in range(num_ancilla):
        a = N + i
        theta = theta_list[i]
        w = wq_list[i]
        wqc.ry(theta, a)
        wqc.cx(a, w)
        wqc.cx(a, w)

    return wqc


def measure_bias(counts: Counter, ancilla_index: int) -> float:
    """
    Compute the empirical probability of measuring '1' on ancilla_index.
    Assumes bitstrings are ordered q_{num_qubits-1}...q_0.
    """
    total = sum(counts.values())
    ones = sum(
        cnt for bitstr, cnt in counts.items() if bitstr[::-1][ancilla_index] == "1"
    )
    return ones / total


def detect_output_bias(
    qc: QuantumCircuit,
    num_ancilla: int = 1,
    thetas: Union[float, List[float]] = np.pi / 6,
    shots: int = 5000,
    tol: float = 0.02,
):
    """
    Test and detect the output-bias watermark(s) in the given circuit.

    Inputs:
      - qc: watermarked circuit with ancillas appended
      - num_ancilla: number of ancillas inserted
      - thetas: single float or list of length `num_ancilla`
      - shots: number of simulation shots
      - tol: allowable deviation from expected bias

    Returns:
      (detected_list, bias_orig_list, bias_trans_list, expected_list)
    """
    if isinstance(thetas, list):
        assert len(thetas) == num_ancilla
        theta_list = thetas
    else:
        theta_list = [thetas] * num_ancilla

    detected = []
    bias_orig_list = []
    bias_trans_list = []
    expected_list = []

    for i in range(num_ancilla):
        anc_idx = qc.num_qubits - num_ancilla + i
        theta = theta_list[i]
        expected = np.sin(theta / 2) ** 2
        expected_list.append(expected)
        print(f"\nAncilla {i} (qubit {anc_idx}) expected bias = {expected:.4f}")

        bias_orig = measure_bias(simulate_counts(qc, shots), anc_idx)
        bias_orig_list.append(bias_orig)
        print(f"Observed bias before transpile: {bias_orig:.4f}")

        if i == 0:
            backend = AerSimulator(method="density_matrix")
            transpiled = transpile(qc, backend, optimization_level=2)
            draw(transpiled, "Transpiled Watermarked Circuit")

        bias_trans = measure_bias(simulate_counts(transpiled, shots), anc_idx)
        bias_trans_list.append(bias_trans)
        print(f"Observed bias after transpile:  {bias_trans:.4f}")

        detected.append(abs(bias_trans - expected) <= tol)

    return detected, bias_orig_list, bias_trans_list, expected_list


def classical_fidelity(dist1: Counter, dist2: Counter) -> float:
    """
    Compute the classical fidelity (Bhattacharyya coefficient) between two distributions.
    F = sum_over_x sqrt(p(x) * q(x)), where p and q are normalized frequencies.
    """
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    fidelity = 0.0
    for bitstr, cnt1 in dist1.items():
        p = cnt1 / total1
        q = dist2.get(bitstr, 0) / total2
        fidelity += np.sqrt(p * q)
    return fidelity


def compare_accuracy(
    orig_qc: QuantumCircuit,
    wm_qc: QuantumCircuit,
    num_ancilla: int = 1,
    shots: int = 5000,
) -> float:
    """
    Compare the output distributions of the original and watermarked circuits
    over the work qubits only, returning the classical fidelity between them.

    Steps:
      1) simulate orig_qc to get dist_orig over N bits
      2) simulate wm_qc to get dist_wm over N+num_ancilla bits
      3) marginalize dist_wm by dropping the ancilla bits (leftmost `num_ancilla`)
      4) compute and return fidelity between dist_orig and marginalized dist_wm
    """
    dist_orig = simulate_counts(orig_qc, shots)

    dist_wm_full = simulate_counts(wm_qc, shots)

    marg = Counter()
    for bitstr, cnt in dist_wm_full.items():
        work_str = bitstr[num_ancilla:]
        marg[work_str] += cnt

    F = classical_fidelity(dist_orig, marg)
    print(f"Classical fidelity between original and watermarked: {F:.4f}")
    return F


def generate_dummy_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    """
    Generate a random test circuit on `num_qubits` qubits with given depth.
    No measurements are added here; they will be auto-injected.
    """
    return random_circuit(num_qubits, depth, max_operands=2)


def main():
    num_qubits = 5
    depth = 6
    num_ancilla = 2
    thetas = [np.pi / 6, np.pi / 4]
    work_qubits = [3, 1]

    qc = generate_dummy_circuit(num_qubits, depth)
    draw(qc, "Original Dummy Circuit")

    wqc = apply_output_bias_watermark(
        qc, num_ancilla=num_ancilla, thetas=thetas, work_qubits=work_qubits
    )
    draw(wqc, "Watermarked Circuit (pre-transpile)")

    detected, bias_orig_list, bias_trans_list, expected_list = detect_output_bias(
        wqc, num_ancilla=num_ancilla, thetas=thetas, shots=3000, tol=0.02
    )
    print(f"\nDetection results per ancilla: {detected}")

    fidelity = compare_accuracy(qc, wqc, num_ancilla=num_ancilla, shots=3000)
    print(f"Accuracy drop due to watermark: {1 - fidelity:.4f}")


if __name__ == "__main__":
    main()
