from utils.qiskit_runtime import register_ibm_runtime, get_backend

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def main():
    try:
        service = register_ibm_runtime()
        print(f"{GREEN}register_ibm_runtime() succeeded.{RESET}")
    except Exception as e:
        print(f"{RED}register_ibm_runtime() failed:{RESET}", e)
        return

    try:
        all_b = service.backends()
        print(f"Found {len(all_b)} backends")
        for b in all_b[:5]:
            print("  -", b.name)
    except Exception as e:
        print(f"{RED}Failed to list backends:{RESET}", e)
        return

    try:
        if any(b.name == "ibm_brisbane" for b in all_b):
            backend = get_backend(service, name="ibm_brisbane")
            print(
                f"{GREEN}get_backend(service, name='ibm_brisbane') succeeded:{RESET}",
                backend.name,
            )
        else:
            backend = get_backend(service)
            print(f"{GREEN}get_backend(service) succeeded:{RESET}", backend.name)
    except Exception as e:
        print(f"{RED}get_backend(...) failed:{RESET}", e)
        return

    print("All checks passed.")


if __name__ == "__main__":
    main()
