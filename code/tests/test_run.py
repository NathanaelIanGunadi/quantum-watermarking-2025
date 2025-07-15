import time
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler

from quantum_watermarking.encoding_patterns import pattern_hh
from quantum_watermarking.embed_watermark import embed_watermark_before_measure

from utils.create_test_circuit import create_test_circuit
from utils.qiskit_runtime import register_ibm_runtime, get_backend


def finalize_with_measure(base_qc: QuantumCircuit) -> QuantumCircuit:
    qc = base_qc.copy()
    qc.measure(qc.qubits[0], qc.clbits[0])
    return qc


def compute_fidelity(
    qc: QuantumCircuit, backend, shots: int = 1024
) -> (float, QuantumCircuit):
    t0 = time.time()
    transpiled_qc = transpile(qc, backend, optimization_level=0)
    t1 = time.time()

    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled_qc], shots=shots)
    print(f"  Transpile time: {t1 - t0:.3f}s")

    result_list = job.result()
    pub_result = result_list[0]
    counts = pub_result.data.measurement.get_counts()
    print(f"  Result counts: {counts}")

    ideal = shots / 2
    fidelity = sum(min(counts.get(bit, 0), ideal) for bit in ["0", "1"]) / shots
    return fidelity, transpiled_qc


def evaluate_original(backend, shots: int):
    base_unitary = create_test_circuit()
    base_qc = finalize_with_measure(base_unitary)
    print("Running Original circuit:")
    fidelity = compute_fidelity(base_qc, backend, shots)
    print(f"  Fidelity (Original): {fidelity}")
    return fidelity


def evaluate_watermarked(backend, shots: int):
    base_unitary = create_test_circuit()
    hh_pattern = pattern_hh(0)
    wm_qc = embed_watermark_before_measure(base_unitary, hh_pattern)
    print("Running Watermarked circuit:")
    fidelity = compute_fidelity(wm_qc, backend, shots)
    print(f"  Fidelity (Watermarked): {fidelity}")
    return fidelity


def run_comparison(shots: int = 1024):
    service = register_ibm_runtime()
    backend = get_backend(service)
    print(f"  Using backend: {backend.name}\n")

    fidelity_orig, _ = evaluate_original(backend, shots)
    fidelity_wm, _ = evaluate_watermarked(backend, shots)

    labels = ["Original", "Watermarked"]
    values = [fidelity_orig, fidelity_wm]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["skyblue", "salmon"])
    plt.ylim(0, 1)
    plt.ylabel("Fidelity (vs. ideal 50/50)")
    plt.title("Fidelity: Original vs Watermarked")

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_comparison()
