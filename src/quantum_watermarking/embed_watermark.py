from qiskit import QuantumCircuit


def embed_watermark_before_measure(
    base_circuit: QuantumCircuit, watermark_circ: QuantumCircuit
) -> QuantumCircuit:
    qc = base_circuit.copy()
    q0 = qc.qubits[0]
    qc.barrier(q0)
    qc.compose(watermark_circ, inplace=True)
    qc.barrier(q0)
    qc.compose(watermark_circ.inverse(), inplace=True)
    qc.barrier(q0)
    qc.measure(q0, qc.clbits[0])
    return qc
