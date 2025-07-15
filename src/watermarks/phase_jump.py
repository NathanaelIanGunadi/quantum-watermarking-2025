import numpy as np
from qiskit import QuantumCircuit
from utils.remove_swaps import remove_swaps


def insert_phase_jump_watermark(
    qc: QuantumCircuit, qubits: list[int], theta: float = np.pi / 7
) -> None:
    """
    In-place: on each qubit in `qubits`, insert RZ(theta); barrier; RZ(-theta).
    """
    for q in qubits:
        qc.barrier()
        qc.rz(theta, q)
        qc.barrier()
        qc.rz(-theta, q)
        qc.barrier()


def detect_phase_jump_watermark(
    transpiled_qc, qubits: list[int], theta: float = np.pi / 7, tol: float = 1e-6
) -> list[int]:
    """
    Returns the subset of `qubits` where the RZ(theta)->...->RZ(-theta)
    pattern is still found after SWAP-removal.
    """
    logical = remove_swaps(transpiled_qc)
    seq = [
        (instr.params[0], qargs[0]._index)
        for instr, qargs, _ in logical.data
        if instr.name == "rz"
    ]
    found = []
    for (angle1, q1), (angle2, q2) in zip(seq, seq[1:]):
        if (
            q1 == q2
            and abs(angle1 - theta) < tol
            and abs(angle2 + theta) < tol
            and q1 in qubits
        ):
            found.append(q1)
    return sorted(set(found))
