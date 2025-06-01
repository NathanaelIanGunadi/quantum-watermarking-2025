from qiskit_ibm_runtime import QiskitRuntimeService
from quantum_watermarking.util import register_ibm_runtime, get_backend


def main():
    try:
        service = register_ibm_runtime()
        print("✅ register_ibm_runtime() succeeded.")
    except Exception as e:
        print("❌ register_ibm_runtime() failed:", e)
        return

    try:
        all_b = service.backends()
        print(f"Found {len(all_b)} backends. Example names:")
        for b in all_b[:5]:
            print("  -", b.name)
    except Exception as e:
        print("❌ Failed to list backends:", e)
        return

    try:
        if any(b.name == "ibm_qasm_simulator" for b in all_b):
            backend = get_backend(service, name="ibm_qasm_simulator")
            print(
                "✅ get_backend(service, name='ibm_qasm_simulator') succeeded:",
                backend.name,
            )
        else:
            backend = get_backend(service)
            print("✅ get_backend(service) succeeded:", backend.name)
    except Exception as e:
        print("❌ get_backend(...) failed:", e)
        return

    print("All checks passed.")


if __name__ == "__main__":
    main()
