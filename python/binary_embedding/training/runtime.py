"""Runtime / numerics helpers: precision, compile, determinism.

`apply_runtime(cfg)` runs once at trainer startup. Keep it side-effect-free
beyond what the user asked for: no global state mutation if cfg=defaults.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from binary_embedding.training.config import RuntimeCfg

log = logging.getLogger(__name__)


def apply_runtime(cfg: RuntimeCfg, *, seed: int) -> None:
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cudnn_benchmark and not cfg.deterministic:
        torch.backends.cudnn.benchmark = True
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        log.info("determinism enabled (FA/SDPA fast paths likely disabled)")
    seed_everything(seed)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(base_seed: int):
    """DataLoader worker_init_fn: seed python+numpy+torch in each worker.

    Each worker `w` gets `seed_everything(base_seed + 1_000_003 * w)` so its
    PRNGs are independent of every other worker's. Pass to DataLoader as
    `worker_init_fn=make_worker_init_fn(seed)`.
    """

    def _init(worker_id: int) -> None:
        seed_everything(base_seed + 1_000_003 * worker_id)

    return _init


def maybe_compile(model: torch.nn.Module, mode: str) -> torch.nn.Module:
    """Wrap `torch.compile`. On compile-time error, log a warning and continue eager.

    `torch.compile` failures are common (kernel mismatches, unsupported ops on
    older drivers, dynamo guard issues). We never want them to brick a run —
    eager fallback is an acceptable performance hit, but losing a 12-hour
    pretraining run because dynamo refused our model is not.
    """
    if mode == "none":
        return model
    try:
        log.info("torch.compile(mode=%s)", mode)
        return torch.compile(model, mode=mode)
    except Exception as exc:  # pragma: no cover - hardware-specific
        log.warning(
            "torch.compile(mode=%s) failed (%s); continuing in eager mode.",
            mode, exc,
        )
        return model


def autocast_context(precision: str, device_type: str):
    """Return a `torch.autocast(...)`-style context manager (no-op for fp32)."""

    class _NoOp:
        def __enter__(self): return None
        def __exit__(self, *a): return False

    if precision == "fp32":
        return _NoOp()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


__all__ = [
    "apply_runtime", "autocast_context", "make_worker_init_fn",
    "maybe_compile", "seed_everything",
]
