# classical_hash_watermark.py

import hashlib
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps  # new API for OpenQASM 2 export


def compute_circuit_hash(qc: QuantumCircuit) -> str:
    # Export to OpenQASM 2 string, then hash its UTF-8 bytes
    qasm_str = dumps(qc).encode("utf-8")
    return hashlib.sha256(qasm_str).hexdigest()


def main():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)

    print("\n=== Circuit to Hash ===")
    print(qc.draw(output="text"))

    digest = compute_circuit_hash(qc)
    print("\nSHA-256 digest:", digest)

    with open("reference_hash.txt", "w") as f:
        f.write(digest)
    print("Reference hash saved to reference_hash.txt")


if __name__ == "__main__":
    main()
