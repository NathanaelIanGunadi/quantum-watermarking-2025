from qiskit import QuantumCircuit


def pattern_hh(qubit_index):
    qc = QuantumCircuit(1, name="H‍H_identity")
    qc.h(0)
    qc.h(0)
    qc.name = f"H‍H_{qubit_index}"
    return qc
