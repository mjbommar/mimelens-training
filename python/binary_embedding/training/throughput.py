"""Throughput counters reported each log-cadence.

Tracks: tokens/s, bytes/s (for byte-budget runs), wall-clock, dataloader-wait %,
host-to-device-copy %, MFU estimate (FLOPs/s ÷ peak-FLOPs given a hardcoded
device peak).
"""

from __future__ import annotations

import dataclasses as dc
import time

import torch


# Conservative peak (bf16/fp16 TFLOPs) for common consumer GPUs.
# Override via env var BINARY_EMBEDDING_PEAK_TFLOPS for unknown devices.
_PEAK_TFLOPS_TABLE: dict[str, float] = {
    "NVIDIA GeForce RTX 4060 Ti": 22.0,
    "NVIDIA GeForce RTX 4090": 165.2,
    "NVIDIA H100": 989.0,
    "NVIDIA A100": 312.0,
    "NVIDIA L40S": 366.0,
}


def device_peak_tflops() -> float:
    name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else ""
    )
    return _PEAK_TFLOPS_TABLE.get(name, float("nan"))


@dc.dataclass(slots=True)
class ThroughputState:
    last_t: float = 0.0
    bytes_per_step: int = 0
    tokens_per_step: int = 0
    flops_per_step: float = 0.0
    accum_dataloader_s: float = 0.0
    accum_h2d_s: float = 0.0
    n_steps: int = 0


class ThroughputMeter:
    """Accumulate per-step costs; emit a metrics dict each `report()`."""

    def __init__(self, *, bytes_per_step: int, tokens_per_step: int, flops_per_step: float) -> None:
        self.s = ThroughputState(
            last_t=time.perf_counter(),
            bytes_per_step=bytes_per_step,
            tokens_per_step=tokens_per_step,
            flops_per_step=flops_per_step,
        )
        self._dl_t0: float | None = None
        self._h2d_t0: float | None = None

    def step(self) -> None:
        self.s.n_steps += 1

    # --- timing windows for sub-phase costs ---

    def begin_dataloader(self) -> None:
        self._dl_t0 = time.perf_counter()

    def end_dataloader(self) -> None:
        if self._dl_t0 is not None:
            self.s.accum_dataloader_s += time.perf_counter() - self._dl_t0
            self._dl_t0 = None

    def begin_h2d(self) -> None:
        self._h2d_t0 = time.perf_counter()

    def end_h2d(self) -> None:
        if self._h2d_t0 is not None:
            self.s.accum_h2d_s += time.perf_counter() - self._h2d_t0
            self._h2d_t0 = None

    def report(self) -> dict[str, float]:
        now = time.perf_counter()
        dt = max(1e-9, now - self.s.last_t)
        steps = max(1, self.s.n_steps)
        ips = steps / dt
        out = {
            "throughput/steps_per_s": ips,
            "throughput/tokens_per_s": self.s.tokens_per_step * ips,
            "throughput/bytes_per_s": self.s.bytes_per_step * ips,
            "throughput/wall_per_step_ms": 1000.0 * dt / steps,
            "throughput/dataloader_pct": 100.0 * self.s.accum_dataloader_s / dt,
            "throughput/h2d_pct": 100.0 * self.s.accum_h2d_s / dt,
        }
        peak = device_peak_tflops()
        if peak == peak and self.s.flops_per_step > 0:  # not NaN
            mfu = (self.s.flops_per_step * ips) / (peak * 1e12)
            out["throughput/mfu_pct"] = 100.0 * mfu

        # reset accumulators
        self.s.last_t = now
        self.s.n_steps = 0
        self.s.accum_dataloader_s = 0.0
        self.s.accum_h2d_s = 0.0
        return out


def estimate_flops_per_step(num_params: int, seq_len: int, batch: int) -> float:
    """Crude 6N estimator from PaLM § A.2 — forward+backward FLOPs per token."""
    return 6.0 * num_params * seq_len * batch


__all__ = [
    "ThroughputMeter", "ThroughputState", "device_peak_tflops", "estimate_flops_per_step",
]
