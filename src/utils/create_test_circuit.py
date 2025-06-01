from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def create_test_circuit() -> QuantumCircuit:
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "measurement")
    qc = QuantumCircuit(q, c)

    qc.h(q[0])
    qc.x(q[0])

    return qc
