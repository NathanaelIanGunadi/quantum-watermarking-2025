"""
ancilla_marking_watermark.py

Entangle each ancilla with two work qubits once, then later detect
those H/CX patterns in the transpiled circuit.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def draw(qc, title):
    # Render the circuit diagram with Matplotlib
    fig = qc.draw(output="mpl", plot_barriers=True)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def insert_ancilla_marking(qc, num_ancilla, work_indices):
    for i in range(num_ancilla):
        a = i
        w = work_indices[i % len(work_indices)]
        qc.h(a)
        qc.cx(a, w)
        qc.cx(a, w)


def detect_ancilla_marking(transpiled_qc, num_ancilla):
    seen_h = set()
    seen_cx = set()

    for instr, qargs, _ in transpiled_qc.data:
        phys = [transpiled_qc.find_bit(q).index for q in qargs]
        if instr.name == "h" and phys[0] < num_ancilla:
            seen_h.add(phys[0])
        if instr.name == "cx" and phys[0] < num_ancilla and phys[1] >= num_ancilla:
            seen_cx.add(phys[0])

    return all(i in seen_h and i in seen_cx for i in range(num_ancilla))


def main():
    backend = AerSimulator()
    qc = QuantumCircuit(5)

    insert_ancilla_marking(qc, num_ancilla=1, work_indices=[1, 3, 4])
    draw(qc, "Original Ancilla-Mark Circuit")

    transpiled = transpile(qc, backend, basis_gates=["h", "cx"], optimization_level=1)
    draw(transpiled, "Transpiled Ancilla-Mark Circuit")

    ok = detect_ancilla_marking(transpiled, num_ancilla=2)
    print("\nAncilla-based watermark survives?", ok)


if __name__ == "__main__":
    main()
