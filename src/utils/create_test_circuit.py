from matplotlib import pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def draw(qc: QuantumCircuit, title: str):
    """Draw the given circuit with Matplotlib."""
    fig = qc.draw(output="mpl", plot_barriers=True)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def create_test_circuit() -> QuantumCircuit:
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "measurement")
    qc = QuantumCircuit(q, c)

    qc.h(q[0])
    qc.x(q[0])

    return qc


def create_2q_test_circuit() -> QuantumCircuit:
    q = QuantumRegister(2, "q")
    c = ClassicalRegister(2, "measurement")
    qc = QuantumCircuit(q, c)

    qc.h(q[0])
    qc.cx(q[0], q[1])

    qc.measure(q, c)

    return qc


if __name__ == "__main__":
    qc = create_2q_test_circuit()
    draw(qc, "test circuit")
