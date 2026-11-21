"""CPU resources shared by the inference backends."""

import os


def available_cpu_count() -> int:
    """Return available logical CPUs, respecting process affinity where supported."""
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return int(process_cpu_count() or 1)
    return os.cpu_count() or 1


THREADS_PER_CT2_WORKER = 4


def configure_torch_cpu_threads() -> None:
    """Set PyTorch's CPU intra-op thread pool to the available processor count."""
    import torch

    torch.set_num_threads(available_cpu_count())


def ctranslate2_cpu_parallelism(
    total_cpus: int | None = None,
    *,
    threads_per_worker: int = THREADS_PER_CT2_WORKER,
) -> tuple[int, int]:
    """Return ``(num_workers, cpu_threads)`` using a fixed threads-per-worker target."""
    cpus = max(1, available_cpu_count() if total_cpus is None else int(total_cpus))
    per_worker = max(1, min(cpus, int(threads_per_worker)))
    num_workers = max(1, cpus // per_worker)
    return num_workers, per_worker
