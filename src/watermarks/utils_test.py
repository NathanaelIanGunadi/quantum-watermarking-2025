import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Dict, Iterable, Tuple, List

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.random import random_circuit

# ---------- circuit builders / drawing ----------


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


def create_random_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    return random_circuit(num_qubits, depth, max_operands=2)


def draw(qc: QuantumCircuit, title: str):
    fig = qc.draw(output="mpl", plot_barriers=True)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------- Aer execution ----------


def simulate_counts(qc: QuantumCircuit, shots: int = 5000) -> Counter:
    backend = AerSimulator(method="density_matrix")
    N = qc.num_qubits
    circ = QuantumCircuit(N, N)
    for instr, qargs, _ in qc.data:
        if instr.name == "measure":
            continue
        new_qargs = [circ.qubits[q._index] for q in qargs]
        circ.append(instr, new_qargs, [])
    circ.measure(range(N), range(N))
    t_qc = transpile(circ, backend, optimization_level=2)
    job = backend.run(t_qc, shots=shots)
    return Counter(job.result().get_counts())


# ---------- metrics ----------


def probs_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tot = sum(counts.values()) or 1
    return {k: v / tot for k, v in counts.items()}


def tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def pst_from_counts(counts: Dict[str, int], n_qubits: int) -> float:
    initial = "0" * n_qubits
    tot = sum(counts.values()) or 1
    return counts.get(initial, 0) / tot


TWO_Q_NAMES = {"cx", "cz", "ecr", "iswap", "xx", "yy", "zz", "rxx", "ryy", "rzz", "dcx"}


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    cnt = 0
    for instr, *_ in qc.data:
        if instr.num_qubits == 2 or instr.name in TWO_Q_NAMES:
            cnt += 1
    return cnt


# ---------- marginalization: drop a bit (little-endian position) ----------


def _drop_bit_little_endian(bitstring: str, pos: int) -> str:
    """
    pos=0 => least-significant bit (rightmost visually).
    We remove that bit and keep orientation as original (MSB...LSB).
    """
    rev = bitstring[::-1]
    trimmed = rev[:pos] + rev[pos + 1 :]
    return trimmed[::-1]


def marginalize_drop_pos(
    counts: Dict[str, int], width: int, drop_pos: int
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for bs, c in counts.items():
        if len(bs) < width:
            bs = bs.zfill(width)
        key = _drop_bit_little_endian(bs, drop_pos)
        out[key] = out.get(key, 0) + c
    return out


# ---------- plotting (single-plot, no specific colors) ----------


def plot_tvd_phase(phases: Iterable[float], tvds: Iterable[float], title: str):
    plt.figure()
    plt.plot(list(phases), list(tvds), marker="o")
    plt.xlabel("Phase (radians)")
    plt.ylabel("TVD")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bar_pairs(
    labels: List[str],
    vals_a: List[float],
    vals_b: List[float],
    ylabel: str,
    legend_a: str,
    legend_b: str,
    title: str,
):
    x = np.arange(len(labels))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, vals_a, width, label=legend_a)
    plt.bar(x + width / 2, vals_b, width, label=legend_b)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
