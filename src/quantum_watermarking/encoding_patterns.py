import numpy as np
from qiskit import QuantumCircuit


def pattern_hh(qubit_index):
    qc = QuantumCircuit(1, name="H‍H_identity")
    qc.h(0)
    qc.h(0)
    qc.name = f"H‍H_{qubit_index}"
    return qc


def pattern_identity_3qbit(qubit_index: int) -> QuantumCircuit:
    qc = QuantumCircuit(1, 0)
    qc.rz(np.pi / 2, qubit_index)
    qc.rx(np.pi, qubit_index)
    # qc.rz(-np.pi / 2, qubit_index)
    return qc


def pattern_identity_2qbit(control: int, target: int) -> QuantumCircuit:
    qc = QuantumCircuit(2, name="2q_identity")
    qc.cx(control, target)
    qc.rz(2 * np.pi / 5, target)
    qc.rx(np.pi / 3, target)
    qc.cx(control, target)
    qc.rx(-np.pi / 3, target)
    qc.rz(-2 * np.pi / 5, target)
    qc.name = f"2q_id_{control}_{target}"
    return qc
