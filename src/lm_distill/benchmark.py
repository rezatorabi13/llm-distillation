"""Model size and CPU inference latency."""

from __future__ import annotations

import copy
import io
import statistics
import time

import torch


def size_mb(model: torch.nn.Module) -> float:
    """Size of the serialized fp32 ``state_dict`` in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1e6


@torch.no_grad()
def cpu_latency_ms(model: torch.nn.Module, batch: torch.Tensor, warmup: int = 3, repeats: int = 20) -> float:
    """Median wall-clock time of one forward pass on CPU, in milliseconds."""
    cpu_model = copy.deepcopy(model).cpu().float().eval()
    batch = batch.cpu()
    for _ in range(warmup):
        cpu_model(batch)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        cpu_model(batch)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings) * 1000
