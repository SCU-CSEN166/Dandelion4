"""CPU resources shared by the inference backends."""

import os


def available_cpu_count() -> int:
    """Return available logical CPUs, respecting process affinity where supported."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return int(process_cpu_count() or 1)
    return os.cpu_count() or 1


def configure_torch_cpu_threads() -> None:
    """Set PyTorch's CPU intra-op thread pool to the available processor count."""
    import torch

    torch.set_num_threads(available_cpu_count())
