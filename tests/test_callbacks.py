"""Callback hooks: ConsoleLogger, JsonlLogger, Checkpointer best-tracking."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from binary_embedding.training.callbacks import (
    Checkpointer,
    ConsoleLogger,
    JsonlLogger,
    TrainContext,
    TrainEvent,
)


def _ctx(out: Path) -> TrainContext:
    return TrainContext(
        out_dir=out, run_id="rid", is_main=True,
        model=nn.Linear(4, 4),
    )


def test_jsonl_logger_writes_events(tmp_path: Path) -> None:
    cb = JsonlLogger()
    ctx = _ctx(tmp_path)
    cb.on_train_start(ctx)
    cb.on_step_end(ctx, TrainEvent(step=0, metrics={"a": 1.0}))
    cb.on_step_end(ctx, TrainEvent(step=1, metrics={"a": 2.0, "b": "x"}))
    cb.on_eval_end(ctx, TrainEvent(step=1, metrics={"eval/x": 0.5}))
    cb.on_train_end(ctx)
    log = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    rows = [json.loads(l) for l in log]
    kinds = [r["kind"] for r in rows]
    assert kinds == ["start", "step", "step", "eval", "end"]
    assert rows[1]["a"] == 1.0


def test_checkpointer_saves_best_on_eval(tmp_path: Path) -> None:
    ckpt = Checkpointer(save_every=0, track_metric="loss", track_mode="min")
    ctx = _ctx(tmp_path)
    ctx.extra["_other_callbacks"] = []
    ckpt.on_eval_end(ctx, TrainEvent(step=10, metrics={"loss": 1.0}))
    assert (tmp_path / "checkpoints" / "best.safetensors").is_file()
    # A worse metric does not overwrite best.
    older_mtime = (tmp_path / "checkpoints" / "best.safetensors").stat().st_mtime
    ckpt.on_eval_end(ctx, TrainEvent(step=20, metrics={"loss": 2.0}))
    assert (tmp_path / "checkpoints" / "best.safetensors").stat().st_mtime == older_mtime
    # A better one does.
    ckpt.on_eval_end(ctx, TrainEvent(step=30, metrics={"loss": 0.5}))
    assert (tmp_path / "checkpoints" / "best.safetensors").stat().st_mtime > older_mtime


def test_checkpointer_keep_last_k(tmp_path: Path) -> None:
    ckpt = Checkpointer(save_every=10, keep_last_k=2)
    ctx = _ctx(tmp_path)
    ctx.extra["_other_callbacks"] = []
    for step in (10, 20, 30, 40):
        ckpt.on_step_end(ctx, TrainEvent(step=step, metrics={}))
    saved = sorted((tmp_path / "checkpoints").glob("step_*.safetensors"))
    assert len(saved) == 2
    assert saved[-1].name.endswith("00000040.safetensors")


def test_console_logger_does_not_crash_on_non_numeric(tmp_path: Path) -> None:
    cb = ConsoleLogger()
    ctx = _ctx(tmp_path)
    cb.on_train_start(ctx)
    cb.on_step_end(ctx, TrainEvent(step=1, metrics={"a": 0.5, "note": "info"}))
    cb.on_train_end(ctx)
