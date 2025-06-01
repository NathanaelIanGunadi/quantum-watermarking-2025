import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()


def register_ibm_runtime() -> QiskitRuntimeService:
    token = os.environ.get("IBMQ_API_TOKEN")
    instance = os.environ.get("IBMQ_INSTANCE_CRN")

    if not token or not instance:
        raise RuntimeError("IBMQ_API_TOKEN or IBMQ_INSTANCE_CRN not found.")

    try:
        QiskitRuntimeService.save_account(
            token=token,
            instance=instance,
            name=os.environ.get("IBMQ_ACCOUNT_NAME", None),
            set_as_default=True,
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise

    service = QiskitRuntimeService(
        instance=instance, name=os.environ.get("IBMQ_ACCOUNT_NAME", None)
    )
    return service


def get_backend(
    service: QiskitRuntimeService,
    name: str = None,
    filters=None,
):
    all_backends = service.backends()
    if name:
        return service.backend(backend_name=name)

    if filters:
        matching = [b for b in all_backends if filters(b)]
    else:
        simulators = [b for b in all_backends if b.name.endswith("simulator")]
        if simulators:
            matching = simulators
        else:
            matching = [b for b in all_backends if b.status().operational]

    if not matching:
        raise RuntimeError("No backends found.")
    return matching[0]
