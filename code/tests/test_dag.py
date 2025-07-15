import argparse

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
from utils.create_test_circuit import create_test_circuit, create_2q_test_circuit

OPTIMIZATON_LEVEL = 1


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
    transpiled_qc = transpile(circuit, backend, optimization_level=OPTIMIZATON_LEVEL)
    dag = circuit_to_dag(transpiled_qc)
    return dag_drawer(dag)


def plot_2q_circuit():
    backend = AerSimulator()

    base_qc = create_2q_test_circuit()
    wm_qc = embed_watermark_before_measure(
        base_qc, pattern_identity_2qbit(control=0, target=1)
    )

    transp_base = transpile(base_qc, backend, optimization_level=OPTIMIZATON_LEVEL)
    transp_water = transpile(wm_qc, backend, optimization_level=OPTIMIZATON_LEVEL)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    base_qc.draw("mpl", idle_wires=False, ax=axes[0, 0])
    axes[0, 0].set_title("Original 2-Qubit Circuit", fontsize=12)
    transp_base.draw("mpl", idle_wires=False, ax=axes[0, 1])
    axes[0, 1].set_title("Transpiled Original", fontsize=12)

    wm_qc.draw("mpl", idle_wires=False, ax=axes[1, 0])
    axes[1, 0].set_title("Watermarked 2-Qubit Circuit", fontsize=12)
    transp_water.draw("mpl", idle_wires=False, ax=axes[1, 1])
    axes[1, 1].set_title("Transpiled Watermarked", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_dag_versions():
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


def plot_circuit_versions():
    backend = AerSimulator()

    orig_qc = build_original_circuit()
    wm_qc = build_watermarked_circuit()

    transpiled_orig = transpile(orig_qc, backend, optimization_level=OPTIMIZATON_LEVEL)
    transpiled_wm = transpile(wm_qc, backend, optimization_level=OPTIMIZATON_LEVEL)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    orig_qc.draw("mpl", idle_wires=False, ax=axes[0, 0])
    axes[0, 0].set_title("Original Circuit (no transpile)", fontsize=14)

    wm_qc.draw("mpl", idle_wires=False, ax=axes[0, 1])
    axes[0, 1].set_title("Watermarked Circuit (no transpile)", fontsize=14)

    transpiled_orig.draw("mpl", idle_wires=False, ax=axes[1, 0])
    axes[1, 0].set_title("Transpiled Original Circuit", fontsize=14)

    transpiled_wm.draw("mpl", idle_wires=False, ax=axes[1, 1])
    axes[1, 1].set_title("Transpiled Watermarked Circuit", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display either circuit diagrams or DAGs for original vs. watermarked circuits"
    )
    parser.add_argument(
        "--mode",
        choices=["circuit", "dag", "two-qubit"],
        required=True,
        help="‘circuit’ to draw circuits, ‘dag’ to draw DAGs",
    )
    args = parser.parse_args()

    if args.mode == "circuit":
        plot_circuit_versions()
    elif args.mode == "two-qubit":
        plot_2q_circuit()
    else:
        plot_dag_versions()
