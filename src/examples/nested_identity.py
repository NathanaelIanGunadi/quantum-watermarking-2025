from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

"""
nested_identity_watermark.py

Insert a 7-gate palindromic identity on qubit pair (a,b) and detect it
directly in the transpiled circuit (no SWAP handling).
"""


def draw(qc, title):
    # Use Matplotlib to render the circuit diagram
    fig = qc.draw(output="mpl", plot_barriers=True)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def insert_nested_identity(qc, pair):
    a, b = pair
    qc.h(a)
    qc.cx(a, b)
    qc.h(a)
    qc.cx(b, a)
    qc.h(a)
    qc.cx(a, b)
    qc.h(a)


def detect_nested_identity(transpiled_qc, pair):
    a, b = pair
    pattern = [
        ("h", (a,)),
        ("cx", tuple(sorted(pair))),
        ("h", (a,)),
        ("cx", tuple(sorted(pair))),
        ("h", (a,)),
        ("cx", tuple(sorted(pair))),
        ("h", (a,)),
    ]
    seq = [
        (instr.name, tuple(sorted(transpiled_qc.find_bit(q).index for q in qargs)))
        for instr, qargs, _ in transpiled_qc.data
    ]
    for i in range(len(seq) - len(pattern) + 1):
        if seq[i : i + len(pattern)] == pattern:
            return True
    return False


def main():
    backend = AerSimulator()
    qc = QuantumCircuit(4)

    insert_nested_identity(qc, (1, 2))
    draw(qc, "Original Nested-Identity Circuit")

    transpiled = transpile(qc, backend, basis_gates=["h", "cx"], optimization_level=2)
    draw(transpiled, "Transpiled Nested-Identity Circuit")

    found = detect_nested_identity(transpiled, (1, 2))
    print("\nNested-identity watermark detected?", found)


if __name__ == "__main__":
    main()
