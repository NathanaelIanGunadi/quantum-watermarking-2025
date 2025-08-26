import os
import time
import logging
from typing import Dict
from dotenv import load_dotenv

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler  # NEW
from qiskit import transpile

# -------- logging --------
LOG_LEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s.%(msecs)03f %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ibm_runtime")

# Load .env (your original style)
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
    log.info(f"[IBM Runtime] Loaded account (instance={instance})")
    return service


def get_backend(
    service: QiskitRuntimeService,
    name: str = None,
    filters=None,
):
    all_backends = service.backends()
    if name:
        be = service.backend(name)
        log.info(f"[IBM Runtime] Backend resolved: {be.name}")
        return be

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
    log.info(f"[IBM Runtime] Auto-selected backend: {matching[0].name}")
    return matching[0]


# ------- helpers (unchanged public API for callers) -------


def _ensure_full_measure(qc):
    """Ensure the circuit measures all qubits to a same-width classical register."""
    if qc.num_clbits >= qc.num_qubits:
        return qc
    from qiskit import QuantumCircuit

    mqc = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    for inst, qargs, _ in qc.data:
        if inst.name != "measure":
            mqc.append(inst, qargs, [])
    mqc.measure(range(qc.num_qubits), range(qc.num_qubits))
    return mqc


def _safe_queue_position(job):
    """Try to read the queue position if exposed (may return None)."""
    try:
        return job.queue_position()
    except Exception:
        return None


def run_on_backend_counts(qc, backend, shots: int = 4096) -> Dict[str, int]:
    """
    Execute on IBM hardware via Qiskit Runtime Primitives V2 in *job mode*
    (no sessions; compatible with Open plan). Returns integer counts.
    """
    mqc = _ensure_full_measure(qc)

    log.info(f"[EXEC] Transpiling for backend={backend.name} (opt=2)…")
    tqc = transpile(mqc, backend=backend, optimization_level=2)
    log.info("[EXEC] Transpile complete.")

    # --- SamplerV2 in *job mode* (no Session) ---
    sampler = Sampler(mode=backend)  # NEW per docs
    log.info(f"[EXEC] Submitting SamplerV2 job (shots={shots})…")
    job = sampler.run([tqc], shots=shots)
    jid = job.job_id()
    log.info(f"[EXEC] Submitted. backend={backend.name} job_id={jid}")

    # Heartbeat
    poll = int(os.getenv("IBM_POLL_SEC", "5"))
    while True:
        try:
            status = str(job.status())
        except Exception as e:
            log.warning(f"[JOB {jid}] status read failed: {e}")
            status = "UNKNOWN"
        log.info(f"[JOB {jid}] status={status}")
        if status.upper() in {"DONE", "CANCELLED", "ERROR"}:
            break
        time.sleep(poll)

    log.info(f"[EXEC] Fetching result for job_id={jid} …")
    result = job.result()  # PrimitiveResult (V2)
    pub_res = result[0]  # first PUB result
    # Combine all registers & get aggregated counts (stable API in V2)
    counts = pub_res.join_data().get_counts()  # NEW
    # Normalize keys + ensure exactly 'shots' total
    width = mqc.num_qubits
    norm_counts: Dict[str, int] = {}
    for k, v in counts.items():
        key = k if isinstance(k, str) else format(k, f"0{width}b")
        norm_counts[key.zfill(width)] = int(v)
    diff = shots - sum(norm_counts.values())
    if diff != 0 and norm_counts:
        top = max(norm_counts.items(), key=lambda kv: kv[1])[0]
        norm_counts[top] += diff
    log.info(f"[EXEC] Done job_id={jid}. Shots={shots}, outcomes={len(norm_counts)}")
    return norm_counts


def transpile_for_backend(qc, backend, optimization_level: int = 2):
    """Transpile helper that mirrors the paper’s assumption (opt level = 2)."""
    return transpile(qc, backend=backend, optimization_level=optimization_level)
