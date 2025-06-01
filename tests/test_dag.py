import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
from qiskit_aer import AerSimulator

from quantum_watermarking.encoding_patterns import (
    pattern_hh,
    pattern_identity_3qbit,
    pattern_identity_2qbit,
)
from quantum_watermarking.embed_watermark import embed_watermark_before_measure
from utils.create_test_circuit import create_test_circuit


def build_original_circuit() -> QuantumCircuit:
    qc = create_test_circuit()
    qc.measure(qc.qubits[0], qc.clbits[0])
    return qc


def build_watermarked_circuit() -> QuantumCircuit:
    base_circ = create_test_circuit()
    wm_qc = embed_watermark_before_measure(base_circ, pattern_identity_3qbit(0))
    return wm_qc


def get_dag_image(circuit: QuantumCircuit) -> any:
    dag = circuit_to_dag(circuit)
    return dag_drawer(dag)


def get_transpiled_dag_image(circuit: QuantumCircuit, backend: AerSimulator) -> any:
    transpiled_qc = transpile(circuit, backend, optimization_level=3)
    dag = circuit_to_dag(transpiled_qc)
    return dag_drawer(dag)


def plot_all_versions():
    backend = AerSimulator()

    orig_qc = build_original_circuit()
    img_orig = get_dag_image(orig_qc)

    wm_qc = build_watermarked_circuit()
    img_wm = get_dag_image(wm_qc)

    img_orig_t = get_transpiled_dag_image(orig_qc, backend)
    img_wm_t = get_transpiled_dag_image(wm_qc, backend)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(img_orig)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original DAG (no transpile)", fontsize=14)

    axes[0, 1].imshow(img_wm)
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Watermarked DAG (no transpile)", fontsize=14)

    axes[1, 0].imshow(img_orig_t)
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Transpiled Original DAG", fontsize=14)

    axes[1, 1].imshow(img_wm_t)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Transpiled Watermarked DAG", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_all_versions()
