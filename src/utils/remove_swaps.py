from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate


def remove_swaps(transpiled_qc: QuantumCircuit) -> QuantumCircuit:
    """
    Undo all SWAP gates by tracking a logical_map.
    Returns a new QuantumCircuit defined on the original logical qubits.
    """
    num_qubits = transpiled_qc.num_qubits
    logical_map = list(range(num_qubits))
    logical_qc = QuantumCircuit(num_qubits, transpiled_qc.num_clbits)

    for instr, qargs, cargs in transpiled_qc.data:
        if instr.name == "swap" or isinstance(instr, SwapGate):
            i = transpiled_qc.find_bit(qargs[0]).index
            j = transpiled_qc.find_bit(qargs[1]).index
            logical_map[i], logical_map[j] = logical_map[j], logical_map[i]
            continue

        # remap all other gates
        mapped = [logical_map[transpiled_qc.find_bit(q).index] for q in qargs]
        logical_qc.append(instr, [logical_qc.qubits[k] for k in mapped], cargs)

    return logical_qc
