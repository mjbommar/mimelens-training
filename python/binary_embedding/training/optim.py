"""Optimizer + param groups + LR schedules.

Three concerns kept separate so any one can be swapped via config:

- `build_param_groups(model, cfg)` — returns a list of optimizer param groups
  with per-group LR multipliers and WD overrides; biases/norms always WD=0;
  optional layer-wise LR decay.
- `build_optimizer(cfg, param_groups)` — picks AdamW / Lion / Adafactor /
  AdamW8bit. Optional deps (Lion, bnb) raise a useful error if missing.
- `build_schedule(cfg.schedule, ..., total_steps, bytes_per_step)` — returns a
  callable `step -> lr_multiplier` for one of cosine / linear / WSD / constant
  with warmup expressible as steps, percentage, or bytes-budget.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from binary_embedding.training.config import (
    AdafactorArgs,
    AdamW8bitArgs,
    AdamWArgs,
    LionArgs,
    OptimCfg,
    ParamGroupRule,
    ParamGroupsCfg,
    ScheduleCfg,
)
from binary_embedding.training.registry import OPTIMIZERS, SCHEDULES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param groups
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _PGroup:
    name: str
    params: list[nn.Parameter]
    lr_multiplier: float
    weight_decay: float


def _llrd_layer_index(name: str, pattern: str) -> int | None:
    m = re.search(pattern, name)
    if m and m.groups():
        try:
            return int(m.group(1))
        except (ValueError, IndexError):
            return None
    return None


def build_param_groups(
    model: nn.Module, cfg: OptimCfg
) -> tuple[list[dict[str, Any]], list[_PGroup]]:
    """Build pytorch-style param groups.

    Returns `(groups_for_optimizer, groups_for_logging)`. The first goes
    straight into `Adam.__init__`; the second is the same data with structured
    metadata used for logging and tests.
    """
    pg_cfg: ParamGroupsCfg = cfg.param_groups
    no_decay_re = re.compile(pg_cfg.no_decay_pattern)
    user_rules = [(re.compile(r.pattern), r) for r in pg_cfg.custom_rules]

    # Determine LLRD denominators if requested.
    layer_indices: dict[str, int] = {}
    max_layer = -1
    if pg_cfg.llrd_decay is not None:
        for n, _ in model.named_parameters():
            li = _llrd_layer_index(n, pg_cfg.llrd_layer_pattern)
            if li is not None:
                layer_indices[n] = li
                if li > max_layer:
                    max_layer = li

    # Bucket each parameter into a (name, lr_multiplier, weight_decay) tuple.
    buckets: dict[tuple[str, float, float], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Default WD: cfg.weight_decay; zero if name matches no_decay regex.
        wd = 0.0 if no_decay_re.search(name) else cfg.weight_decay
        lr_mult = 1.0
        bucket_label = "decay" if wd > 0 else "no_decay"

        # User custom rules can override WD and scale LR.
        for rx, rule in user_rules:
            if rx.search(name):
                lr_mult *= rule.lr_multiplier
                if rule.weight_decay is not None:
                    wd = rule.weight_decay
                bucket_label = rule.name
                break

        # LLRD: deeper layers get a smaller LR multiplier.
        if pg_cfg.llrd_decay is not None and name in layer_indices:
            li = layer_indices[name]
            depth_from_top = max_layer - li
            lr_mult *= pg_cfg.llrd_decay ** depth_from_top
            bucket_label = f"{bucket_label}.layer{li}"

        key = (bucket_label, round(lr_mult, 8), round(wd, 8))
        buckets.setdefault(key, []).append(param)

    pgroups_for_logging: list[_PGroup] = []
    pgroups_for_optim: list[dict[str, Any]] = []
    for (label, lr_mult, wd), params in sorted(buckets.items(), key=lambda x: x[0][0]):
        pgroups_for_logging.append(
            _PGroup(name=label, params=params, lr_multiplier=lr_mult, weight_decay=wd)
        )
        pgroups_for_optim.append(
            {
                "params": params,
                "lr": cfg.lr * lr_mult,
                "weight_decay": wd,
                "_name": label,
                "_lr_multiplier": lr_mult,
            }
        )
    return pgroups_for_optim, pgroups_for_logging


# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------


@OPTIMIZERS.register("adamw")
def _build_adamw(spec: AdamWArgs, params: list[dict]) -> torch.optim.Optimizer:
    fused = spec.fused and torch.cuda.is_available()
    return torch.optim.AdamW(params, betas=spec.betas, eps=spec.eps, fused=fused)


@OPTIMIZERS.register("adamw8bit")
def _build_adamw_8bit(spec: AdamW8bitArgs, params: list[dict]) -> torch.optim.Optimizer:
    try:
        import bitsandbytes as bnb
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "AdamW8bit requested but bitsandbytes is not installed. "
            "uv add bitsandbytes"
        ) from e
    return bnb.optim.AdamW8bit(params, betas=spec.betas, eps=spec.eps)


@OPTIMIZERS.register("lion")
def _build_lion(spec: LionArgs, params: list[dict]) -> torch.optim.Optimizer:
    try:
        from lion_pytorch import Lion
    except ImportError as e:  # pragma: no cover
        raise ImportError("Lion requested but lion-pytorch is not installed.") from e
    return Lion(params, betas=spec.betas)


@OPTIMIZERS.register("adafactor")
def _build_adafactor(spec: AdafactorArgs, params: list[dict]) -> torch.optim.Optimizer:
    # Use HF Transformers' Adafactor (no extra dep — already in our env).
    try:
        from transformers.optimization import Adafactor
    except ImportError as e:  # pragma: no cover
        raise ImportError("Adafactor requested but transformers not installed") from e
    return Adafactor(
        params,
        relative_step=spec.relative_step,
        scale_parameter=spec.scale_parameter,
        warmup_init=False,
    )


def build_optimizer(cfg: OptimCfg, param_groups: list[dict[str, Any]]) -> torch.optim.Optimizer:
    return OPTIMIZERS.build(cfg.optimizer.type, cfg.optimizer, param_groups)


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------


def _resolve_warmup_steps(
    sched: ScheduleCfg, total_steps: int, bytes_per_step: int
) -> int:
    if sched.warmup_steps is not None:
        return max(1, int(sched.warmup_steps))
    if sched.warmup_bytes is not None:
        return max(1, int(sched.warmup_bytes // max(1, bytes_per_step)))
    if sched.warmup_pct is not None:
        return max(1, int(total_steps * sched.warmup_pct))
    raise AssertionError("ScheduleCfg validator should have caught this")


@SCHEDULES.register("cosine")
def _cosine(sched: ScheduleCfg, total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return sched.min_lr_pct + (1 - sched.min_lr_pct) * decay
    return f


@SCHEDULES.register("linear")
def _linear(sched: ScheduleCfg, total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return sched.min_lr_pct + (1 - sched.min_lr_pct) * (1.0 - progress)
    return f


@SCHEDULES.register("constant")
def _constant(sched: ScheduleCfg, total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return f


@SCHEDULES.register("wsd")
def _wsd(sched: ScheduleCfg, total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """Warmup-Stable-Decay (Olmo / WSD): warm up, hold at peak, then decay."""
    stable_pct = sched.stable_pct if sched.stable_pct is not None else 0.7
    stable_end = warmup_steps + int(stable_pct * (total_steps - warmup_steps))

    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < stable_end:
            return 1.0
        progress = (step - stable_end) / max(1, total_steps - stable_end)
        # Linear decay in WSD; literature also uses sqrt or 1/x.
        return sched.min_lr_pct + (1 - sched.min_lr_pct) * (1.0 - progress)
    return f


def build_schedule(
    sched: ScheduleCfg, total_steps: int, bytes_per_step: int = 1
) -> Callable[[int], float]:
    """Return `step -> lr_multiplier` ∈ [0, 1].

    Multiply by each param group's `_lr_multiplier * cfg.optim.lr` to get the
    actual LR. The Trainer applies this at the start of each optimizer step.
    """
    warmup_steps = _resolve_warmup_steps(sched, total_steps, bytes_per_step)
    return SCHEDULES.build(sched.type, sched, total_steps, warmup_steps)


class LRScheduler:
    """Stateful wrapper around the schedule closure.

    Today's schedules are pure functions of `step`, so the only state is the
    counter. The wrapper exists so we can save/restore it across resumes
    (and so future stateful schedules — e.g. ReduceOnPlateau — slot in
    without changing the trainer).
    """

    def __init__(
        self,
        sched_cfg: ScheduleCfg,
        total_steps: int,
        bytes_per_step: int = 1,
    ) -> None:
        self._cfg = sched_cfg
        self._total_steps = int(total_steps)
        self._bytes_per_step = int(bytes_per_step)
        self._fn = build_schedule(sched_cfg, total_steps, bytes_per_step)
        self._step = 0

    @property
    def step_count(self) -> int:
        return self._step

    def lr_multiplier(self, step: int | None = None) -> float:
        s = self._step if step is None else int(step)
        return float(self._fn(s))

    def advance(self) -> None:
        self._step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "total_steps": self._total_steps,
            "bytes_per_step": self._bytes_per_step,
            # Keep cfg.type for sanity check on resume; we don't try to restore
            # the cfg itself — the cfg is rebuilt from the user's YAML.
            "schedule_type": self._cfg.type,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schedule_type") != self._cfg.type:
            log.warning(
                "scheduler resume: type mismatch (was %s, now %s); "
                "step counter will still be restored",
                state.get("schedule_type"), self._cfg.type,
            )
        self._step = int(state.get("step", 0))
        # Note: we do NOT clobber total_steps/bytes_per_step from the saved
        # state — those come from the (potentially updated) cfg on resume.


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ClipResult:
    grad_norm: float
    clipped: bool   # True iff measured grad-norm > by_norm threshold
    skipped: bool   # True iff NaN/Inf detected and grads were zeroed


def clip_gradients(
    parameters: list[nn.Parameter],
    *,
    by_norm: float | None,
    by_value: float | None,
    skip_on_nan: bool,
) -> ClipResult:
    """Apply clip-by-norm and/or clip-by-value.

    Returns a `ClipResult` with three signals:
    - `grad_norm`: pre-clip L2 norm (NaN if no grads or skipped on NaN/Inf).
    - `clipped`: True iff `grad_norm > by_norm` (i.e. clip-by-norm actually scaled).
    - `skipped`: True iff `skip_on_nan=True` and grads contained NaN/Inf (zeroed).
    """
    has_grads = [p for p in parameters if p.grad is not None]
    if not has_grads:
        return ClipResult(grad_norm=float("nan"), clipped=False, skipped=False)
    if skip_on_nan:
        any_nan = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in has_grads
        )
        if any_nan:
            for p in has_grads:
                p.grad = None
            return ClipResult(grad_norm=float("nan"), clipped=False, skipped=True)
    if by_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(has_grads, by_norm))
        was_clipped = grad_norm > by_norm
    else:
        # Just measure for logging.
        with torch.no_grad():
            sq = sum(float(torch.norm(p.grad.float()).item()) ** 2 for p in has_grads)
            grad_norm = math.sqrt(sq)
        was_clipped = False
    if by_value is not None:
        torch.nn.utils.clip_grad_value_(has_grads, by_value)
    return ClipResult(grad_norm=grad_norm, clipped=was_clipped, skipped=False)


__all__ = [
    "ClipResult", "LRScheduler", "build_optimizer", "build_param_groups",
    "build_schedule", "clip_gradients",
]
