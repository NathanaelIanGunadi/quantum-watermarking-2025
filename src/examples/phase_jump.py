import numpy as np
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator


def draw(qc, title):
    # This version returns a matplotlib Figure
    fig = qc.draw(output="mpl", plot_barriers=True)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def insert_phase_jump(qc, qubits, theta=np.pi / 7):
    for q in qubits:
        qc.rz(theta, q)
        qc.barrier()
        qc.rz(-theta, q)


def detect_phase_jump(transpiled_qc, qubits, theta=np.pi / 7, tol=1e-6):
    # unchanged...
    found = []
    ops = [
        (instr.name, transpiled_qc.find_bit(qargs[0]).index, float(instr.params[0]))
        for instr, qargs, _ in transpiled_qc.data
        if instr.name == "rz"
    ]
    for (name1, q1, a1), (name2, q2, a2) in zip(ops, ops[1:]):
        if q1 == q2 == q1 in qubits and abs(a1 - theta) < tol and abs(a2 + theta) < tol:
            found.append(q1)
    return sorted(set(found))


def main():
    backend = AerSimulator()
    qc = QuantumCircuit(4)

    insert_phase_jump(qc, [1, 3])

    draw(qc, "Original Phase-Jump Circuit")
    transpiled = transpile(
        qc, backend, basis_gates=["rz", "cx", "h"], optimization_level=3
    )
    draw(transpiled, "Transpiled Phase-Jump Circuit")

    detected = detect_phase_jump(transpiled, [1, 3])
    print("\nDetected phase-jump watermark on qubits:", detected)


if __name__ == "__main__":
    main()
