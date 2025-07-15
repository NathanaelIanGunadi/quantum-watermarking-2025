from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def create_test_circuit() -> QuantumCircuit:
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "measurement")
    qc = QuantumCircuit(q, c)

    qc.h(q[0])
    qc.x(q[0])

    return qc


def create_2q_test_circuit() -> QuantumCircuit:
    q = QuantumRegister(2, "q")
    c = ClassicalRegister(1, "measurement")
    qc = QuantumCircuit(q, c)

    qc.h(q[0])
    qc.cx(q[0], q[1])

    return qc
