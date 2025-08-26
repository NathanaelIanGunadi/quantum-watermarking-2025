import matplotlib.pyplot as plt
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.random import random_circuit


def generate_dummy_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    """Produce a random circuit on `num_qubits` wires."""
    return random_circuit(num_qubits, depth, max_operands=2)


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
