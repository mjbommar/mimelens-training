"""Callback hooks for the training loop.

A callback is any object implementing the `Callback` protocol. The trainer
fires `on_train_start`, `on_step_end`, `on_eval_end`, `on_checkpoint`,
`on_train_end`. We avoid class hierarchies — duck-typing keeps it simple.

Default callbacks shipped here:
- `ConsoleLogger` — formatted training-log line.
- `JsonlLogger` — one JSON object per event into out_dir/events.jsonl.
- `Checkpointer` — periodic + best-by-metric saves; manages `keep_last_k`.
- `WandbLogger` — optional, only loaded if `cfg.log.wandb_project` is set.
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
from safetensors.torch import save_file


log = logging.getLogger(__name__)


@dc.dataclass(slots=True)
class TrainEvent:
    step: int
    metrics: Mapping[str, float]
    extras: Mapping[str, Any] = dc.field(default_factory=dict)


class Callback(Protocol):
    def on_train_start(self, ctx: "TrainContext") -> None: ...
    def on_step_end(self, ctx: "TrainContext", event: TrainEvent) -> None: ...
    def on_eval_end(self, ctx: "TrainContext", event: TrainEvent) -> None: ...
    def on_checkpoint(self, ctx: "TrainContext", path: Path, kind: str) -> None: ...
    def on_train_end(self, ctx: "TrainContext") -> None: ...


@dc.dataclass(slots=True)
class TrainContext:
    """Live training state passed into every callback hook."""

    out_dir: Path
    run_id: str
    is_main: bool
    model: nn.Module
    optimizer: torch.optim.Optimizer | None = None
    extra: dict[str, Any] = dc.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default callbacks
# ---------------------------------------------------------------------------


class ConsoleLogger:
    """Pretty single-line console logger with selectable metric keys."""

    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = keys
        self._t0 = time.perf_counter()

    def on_train_start(self, ctx: TrainContext) -> None:
        log.info("[%s] training started in %s", ctx.run_id, ctx.out_dir)

    def on_step_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        if not ctx.is_main:
            return
        keys = self.keys or list(ev.metrics)
        body = "  ".join(
            f"{k}={ev.metrics[k]:.4f}"
            for k in keys if k in ev.metrics and isinstance(ev.metrics[k], (int, float))
        )
        log.info("step %6d | %s", ev.step, body)

    def on_eval_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        if not ctx.is_main:
            return
        body = "  ".join(f"{k}={v:.4f}" for k, v in ev.metrics.items()
                         if isinstance(v, (int, float)))
        log.info("eval  %6d | %s", ev.step, body)

    def on_checkpoint(self, ctx: TrainContext, path: Path, kind: str) -> None:
        if ctx.is_main:
            log.info("ckpt(%s) -> %s", kind, path)

    def on_train_end(self, ctx: TrainContext) -> None:
        if ctx.is_main:
            log.info("[%s] training done in %.1fs", ctx.run_id, time.perf_counter() - self._t0)


class JsonlLogger:
    """Append every event to `out_dir/events.jsonl`."""

    def __init__(self) -> None:
        self._fh = None

    def _open(self, ctx: TrainContext) -> None:
        if self._fh is None and ctx.is_main:
            ctx.out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(ctx.out_dir / "events.jsonl", "a", buffering=1)

    def _write(self, kind: str, step: int, payload: Mapping[str, Any]) -> None:
        if self._fh is None:
            return
        rec = {"kind": kind, "step": step, "ts": time.time(),
               **{k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in payload.items()}}
        self._fh.write(json.dumps(rec, default=_jsonable) + "\n")

    def on_train_start(self, ctx: TrainContext) -> None:
        self._open(ctx)
        self._write("start", 0, {"run_id": ctx.run_id})

    def on_step_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        self._open(ctx)
        self._write("step", ev.step, dict(ev.metrics))

    def on_eval_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        self._open(ctx)
        self._write("eval", ev.step, dict(ev.metrics))

    def on_checkpoint(self, ctx: TrainContext, path: Path, kind: str) -> None:
        self._open(ctx)
        self._write("checkpoint", 0, {"path": str(path), "kind": kind})

    def on_train_end(self, ctx: TrainContext) -> None:
        if self._fh is not None:
            self._write("end", 0, {})
            self._fh.close()


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    return repr(v)


class Checkpointer:
    """Periodic + best-by-metric safetensors checkpoints, with keep_last_k pruning."""

    def __init__(
        self,
        *,
        save_every: int,
        keep_last_k: int = 3,
        track_metric: str | None = None,
        track_mode: str = "min",
    ) -> None:
        self.save_every = save_every
        self.keep_last_k = keep_last_k
        self.track_metric = track_metric
        self.track_mode = track_mode
        self._best: float | None = None

    def _ckpt_dir(self, ctx: TrainContext) -> Path:
        d = ctx.out_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save(self, ctx: TrainContext, name: str) -> Path:
        d = self._ckpt_dir(ctx)
        path = d / name
        state = (
            ctx.model.module.state_dict() if hasattr(ctx.model, "module")
            else ctx.model.state_dict()
        )
        save_file(state, str(path))
        return path

    def _prune(self, ctx: TrainContext) -> None:
        if self.keep_last_k <= 0:
            return
        d = self._ckpt_dir(ctx)
        steps = sorted(d.glob("step_*.safetensors"))
        for old in steps[: max(0, len(steps) - self.keep_last_k)]:
            try:
                old.unlink()
            except OSError:
                pass

    def on_train_start(self, ctx: TrainContext) -> None: ...

    def on_step_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        if not ctx.is_main:
            return
        if self.save_every > 0 and ev.step > 0 and ev.step % self.save_every == 0:
            path = self._save(ctx, f"step_{ev.step:08d}.safetensors")
            self._prune(ctx)
            for cb in ctx.extra.get("_other_callbacks", []):
                cb.on_checkpoint(ctx, path, "step")

    def on_eval_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        if not ctx.is_main or self.track_metric is None:
            return
        v = ev.metrics.get(self.track_metric)
        if v is None or not isinstance(v, (int, float)):
            return
        better = (
            (self._best is None)
            or (self.track_mode == "min" and v < self._best)
            or (self.track_mode == "max" and v > self._best)
        )
        if better:
            self._best = float(v)
            path = self._save(ctx, "best.safetensors")
            for cb in ctx.extra.get("_other_callbacks", []):
                cb.on_checkpoint(ctx, path, "best")

    def on_checkpoint(self, ctx: TrainContext, path: Path, kind: str) -> None: ...
    def on_train_end(self, ctx: TrainContext) -> None: ...


class WandbLogger:
    """Optional wandb adapter (only constructed when wandb_project is set)."""

    def __init__(self, project: str, run_name: str | None, run_id: str, config: dict[str, Any]) -> None:
        try:
            import wandb  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("wandb_project set but `wandb` not installed") from e
        self._wandb = __import__("wandb")
        self._inited = False
        self._project = project
        self._run_name = run_name or run_id
        self._config = config
        self._run_id = run_id

    def on_train_start(self, ctx: TrainContext) -> None:
        if not ctx.is_main:
            return
        self._wandb.init(
            project=self._project,
            name=self._run_name,
            id=self._run_id,
            config=self._config,
            dir=str(ctx.out_dir),
            resume="allow",
        )
        self._inited = True

    def on_step_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        if not self._inited or not ctx.is_main:
            return
        self._wandb.log({k: v for k, v in ev.metrics.items()
                         if isinstance(v, (int, float))}, step=ev.step)

    def on_eval_end(self, ctx: TrainContext, ev: TrainEvent) -> None:
        if not self._inited or not ctx.is_main:
            return
        self._wandb.log({k: v for k, v in ev.metrics.items()
                         if isinstance(v, (int, float))}, step=ev.step)

    def on_checkpoint(self, ctx: TrainContext, path: Path, kind: str) -> None: ...

    def on_train_end(self, ctx: TrainContext) -> None:
        if self._inited:
            self._wandb.finish()


__all__ = [
    "Callback", "Checkpointer", "ConsoleLogger", "JsonlLogger",
    "TrainContext", "TrainEvent", "WandbLogger",
]
