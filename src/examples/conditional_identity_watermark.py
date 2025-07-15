# conditional_identity_watermark.py

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.controlflow import IfElseOp


def insert_conditional_watermark(qc, ancilla, thetas):
    """
    Insert a conditional identity watermark on 'ancilla'.
    For each theta in 'thetas':
      - measure ancilla -> classical bit i
      - if that bit == 1: apply RZ(theta) and RZ(-theta) on ancilla
      - reset ancilla
    """
    for i, theta in enumerate(thetas):
        qc.measure(ancilla, i)
        true_branch = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        true_branch.rz(theta, ancilla)
        true_branch.rz(-theta, ancilla)
        false_branch = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        qc.if_else((i, 1), true_branch, false_branch, qc.qubits, qc.clbits)
        qc.reset(ancilla)


def detect_conditional_watermark(transpiled, ancilla, thetas, tol=1e-8):
    """
    Detect the measure->if_else->reset pattern with matching thetas.
    Returns True if all watermark bits are found.
    """
    data = transpiled.data
    idx = 0
    for i, theta in enumerate(thetas):
        # 1) find measure ancilla -> c_i
        found = False
        while idx < len(data):
            instr, qargs, cargs = data[idx]
            if (
                instr.name == "measure"
                and transpiled.find_bit(qargs[0]).index == ancilla
                and cargs[0]._index == i
            ):
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            return False

        # 2) find IfElseOp next
        if idx >= len(data):
            return False
        instr, qargs, cargs = data[idx]
        if not isinstance(instr, IfElseOp):
            return False

        # extract true branch from blocks
        true_block, false_block = instr.blocks
        tb_data = true_block.data
        # expect first two ops RZ(theta), RZ(-theta)
        if len(tb_data) < 2:
            return False
        name1, _, p1 = tb_data[0]
        name2, _, p2 = tb_data[1]
        if name1 != "rz" or name2 != "rz":
            return False
        if not np.isclose(float(p1[0]), theta, atol=tol):
            return False
        if not np.isclose(float(p2[0]), -theta, atol=tol):
            return False
        idx += 1

        # 3) find reset ancilla
        found = False
        while idx < len(data):
            instr, qargs, cargs = data[idx]
            if instr.name == "reset" and transpiled.find_bit(qargs[0]).index == ancilla:
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            return False

    return True


def draw(qc, title):
    print(f"\n=== {title} ===")
    print(qc.draw(output="text"))


def main():
    # watermark bit angles
    thetas = [np.pi / 8, np.pi / 6, np.pi / 4]
    # build circuit: 1 ancilla + 2 work qubits, 3 classical bits for watermark + 2 for work measurement
    qc = QuantumCircuit(3, len(thetas) + 2)

    insert_conditional_watermark(qc, ancilla=0, thetas=thetas)

    # example algorithm on work qubits
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()

    # measure work qubits
    qc.measure(1, len(thetas))
    qc.measure(2, len(thetas) + 1)

    draw(qc, "Original circuit with conditional watermark")

    backend = AerSimulator()
    transpiled = transpile(qc, backend, optimization_level=0)
    draw(transpiled, "Transpiled circuit with conditional watermark")

    found = detect_conditional_watermark(transpiled, ancilla=0, thetas=thetas)
    print("\nConditional watermark detected?", found)


if __name__ == "__main__":
    main()
