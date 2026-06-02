"""Trainer: wires config + encoder + losses + optimizer + schedule + callbacks.

This is the new orchestrator. The old `pretrain.py` becomes a thin CLI shim
delegating to `Trainer.train(cfg)`.

Responsibilities:
- Build the model, the loss (single or composite), optimizer, scheduler, EMA.
- Build the dataloader from the train cache, plus an eval cache for in-loop eval.
- Run the step loop with bf16 autocast, grad accum, grad clip, throughput counters.
- Fire callbacks at start / step / eval / checkpoint / end.
- Auto-resume from `out_dir/checkpoints/state.pt` (full state) if present.
- Dump resolved config + manifest at startup.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from binary_embedding.data.cache import (
    Variant,
    variant_bpe,
    variant_byte,
)
from binary_embedding.data.dataset import MLMTokenStreamDataset, TokenCache
from binary_embedding.models.encoder import BinaryEncoder, EncoderConfig
from binary_embedding.training.callbacks import (
    Callback,
    Checkpointer,
    ConsoleLogger,
    JsonlLogger,
    TrainContext,
    TrainEvent,
    WandbLogger,
)
from binary_embedding.training.config import (
    CompositeLossCfg,
    RunConfig,
    dump_resolved_config,
)
from binary_embedding.training.eval_loop import run_inloop_eval
from binary_embedding.training.losses import LossInputs, build_loss
from binary_embedding.training.manifest import write_manifest
from binary_embedding.training.optim import (
    LRScheduler,
    build_optimizer,
    build_param_groups,
    clip_gradients,
)
from binary_embedding.training.regularization import WeightEMA
from binary_embedding.training.runtime import (
    apply_runtime,
    autocast_context,
    make_worker_init_fn,
    maybe_compile,
)
from binary_embedding.training.throughput import (
    ThroughputMeter,
    estimate_flops_per_step,
)

log = logging.getLogger("trainer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _variant_from_cfg(cfg: RunConfig) -> Variant:
    name = cfg.variant.pipeline
    if name == "byte":
        return variant_byte()
    n = int(name[4:].rstrip("k")) * (1024 if name.endswith("k") else 1)
    return variant_bpe(n)


def _make_encoder(cfg: RunConfig, vocab_size: int) -> BinaryEncoder:
    ec = EncoderConfig(
        vocab_size=vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        ffn_multiplier_num=cfg.model.ffn_multiplier_num,
        ffn_multiplier_den=cfg.model.ffn_multiplier_den,
        max_seq_len=cfg.data.seq_len,
        rope_theta=cfg.model.rope_theta,
        rms_norm_eps=cfg.model.rms_norm_eps,
        cls_pool_dim=cfg.model.cls_pool_dim,
        initializer_range=cfg.reg.init_std,
        init_scheme=cfg.reg.init_scheme,
        embedding_dropout=cfg.reg.embedding_dropout,
        hidden_dropout=cfg.reg.hidden_dropout,
        attention_dropout=cfg.reg.attention_dropout,
        drop_path_rate=cfg.reg.drop_path_rate,
        layer_scale_init=cfg.reg.layer_scale_init,
        grad_checkpointing=cfg.model.grad_checkpointing,
    )
    return BinaryEncoder(ec)


def _build_callbacks(cfg: RunConfig) -> list[Callback]:
    cbs: list[Callback] = [
        ConsoleLogger(),
        JsonlLogger(),
        Checkpointer(
            save_every=cfg.log.save_every,
            keep_last_k=cfg.log.keep_last_k,
            track_metric=cfg.log.track_best_metric,
            track_mode=cfg.log.track_best_mode,
        ),
    ]
    if cfg.log.wandb_project:
        cbs.append(
            WandbLogger(
                project=cfg.log.wandb_project,
                run_name=cfg.log.wandb_run_name,
                run_id=cfg.run_id,
                config={"config_hash": cfg.hash_short()},
            )
        )
    return cbs


# ---------------------------------------------------------------------------
# DDP env detection
# ---------------------------------------------------------------------------


def _init_dist() -> tuple[int, int, int, torch.device]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        world = dist.get_world_size()
        rank = dist.get_rank()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return world, rank, local, torch.device(f"cuda:{local}")
    if torch.cuda.is_available():
        return 1, 0, 0, torch.device("cuda:0")
    return 1, 0, 0, torch.device("cpu")


# ---------------------------------------------------------------------------
# Resume / persist
# ---------------------------------------------------------------------------


def _save_full_state(
    *, path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    ema: WeightEMA | None, scheduler: "LRScheduler | None", step: int,
    rng: dict[str, Any],
) -> None:
    payload = {
        "step": step,
        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **rng,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _load_full_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(str(path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def train(cfg: RunConfig, *, repo_root: Path | None = None) -> None:
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    world, rank, local_rank, device = _init_dist()
    is_main = rank == 0

    apply_runtime(cfg.runtime, seed=cfg.seed + rank)
    out_dir = cfg.log.out_dir
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_resolved_config(cfg, out_dir)
        write_manifest(
            out_dir,
            run_id=cfg.run_id,
            config_hash=cfg.hash_short(),
            config_resolved_path=out_dir / "config.resolved.yaml",
            repo_root=repo_root,
        )

    variant = _variant_from_cfg(cfg)
    train_cache = TokenCache.open(variant, cfg.data.train_split, cfg.data.cache_root)
    eval_cache: TokenCache | None = None
    if cfg.log.eval_every > 0:
        try:
            eval_cache = TokenCache.open(variant, cfg.data.eval_split, cfg.data.cache_root)
        except FileNotFoundError as e:
            if is_main:
                log.warning("eval cache missing — disabling in-loop eval (%s)", e)
            eval_cache = None

    model = _make_encoder(cfg, vocab_size=variant.vocab_size).to(device)
    if is_main:
        log.info(
            "model: total=%d  backbone=%d  variant=%s",
            model.num_parameters(),
            model.num_parameters(exclude_embeddings=True),
            variant.name,
        )

    ema = WeightEMA(model, cfg.reg.ema_decay) if cfg.reg.ema_decay else None
    model = maybe_compile(model, cfg.runtime.compile_mode)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_fn = build_loss(
        cfg.loss,
        hidden_size=cfg.model.hidden_size,
        cls_pool_dim=cfg.model.cls_pool_dim,
    ).to(device)

    pg_for_optim, pg_for_log = build_param_groups(
        model.module if isinstance(model, DDP) else model, cfg.optim
    )
    # Add loss params (head modules) as their own group with full LR + WD.
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if loss_params:
        pg_for_optim.append({
            "params": loss_params, "lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay,
            "_name": "loss_heads", "_lr_multiplier": 1.0,
        })

    optim = build_optimizer(cfg.optim, pg_for_optim)

    # GradScaler for fp16; bf16 and fp32 don't need one.
    scaler: torch.amp.GradScaler | None = None
    if cfg.runtime.precision == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        if is_main:
            log.info("fp16 GradScaler enabled")

    # Compute per-step bytes for byte-budget scheduling.
    micro = cfg.batching.per_gpu_batch
    accum = cfg.batching.grad_accum
    seq_len = cfg.data.seq_len
    tokens_per_step = micro * accum * seq_len * world
    if variant.is_byte:
        bytes_per_step = tokens_per_step
    else:
        avg = float(train_cache.n_bytes.sum()) / max(1, int(train_cache.n_tokens.sum()))
        bytes_per_step = int(tokens_per_step * avg)

    sb = cfg.schedule_budget
    if sb.target_steps is not None:
        total_steps = int(sb.target_steps)
    elif sb.target_bytes is not None:
        total_steps = max(1, sb.target_bytes // bytes_per_step)
    else:
        assert sb.target_tokens is not None
        total_steps = max(1, sb.target_tokens // tokens_per_step)

    scheduler = LRScheduler(cfg.schedule, total_steps, bytes_per_step)
    if is_main:
        log.info(
            "schedule total_steps=%d bytes/step=%s tokens/step=%s",
            total_steps, f"{bytes_per_step:,}", f"{tokens_per_step:,}"
        )

    # Dataloader
    train_ds = MLMTokenStreamDataset(
        train_cache, seq_len=seq_len, mask_ratio=cfg.data.mask_ratio,
        weighted_by_bytes=cfg.data.doc_sampling == "sqrt_bytes",
        seed=cfg.seed + rank,
    )
    loader = DataLoader(
        train_ds, batch_size=micro, num_workers=cfg.batching.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.batching.num_workers > 0,
        drop_last=True,
        worker_init_fn=make_worker_init_fn(cfg.seed + rank),
    )

    # Throughput
    backbone_params = (model.module if isinstance(model, DDP) else model).num_parameters(
        exclude_embeddings=False
    )
    flops_per_step = estimate_flops_per_step(backbone_params, seq_len, micro * accum * world)
    meter = ThroughputMeter(
        bytes_per_step=bytes_per_step, tokens_per_step=tokens_per_step, flops_per_step=flops_per_step
    )

    # Callbacks
    callbacks = _build_callbacks(cfg)
    ctx = TrainContext(
        out_dir=out_dir, run_id=cfg.run_id, is_main=is_main,
        model=model, optimizer=optim, extra={"_other_callbacks": callbacks},
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    # Auto-resume
    state_path = out_dir / "checkpoints" / "state.pt"
    start_step = 0
    state = _load_full_state(state_path)
    if state is not None:
        if is_main:
            log.info("resuming from step %d", state["step"])
        (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("torch_cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
        start_step = int(state["step"])

    iterator: Iterator[dict[str, torch.Tensor]] = iter(loader)
    train_t0 = time.perf_counter()

    for step in range(start_step, total_steps):
        lr_mult = scheduler.lr_multiplier(step)
        for pg in optim.param_groups:
            base_mult = pg.get("_lr_multiplier", 1.0)
            pg["lr"] = cfg.optim.lr * base_mult * lr_mult

        per_step_metrics: dict[str, float] = {}
        loss_val = 0.0
        for _ in range(accum):
            meter.begin_dataloader()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            meter.end_dataloader()

            meter.begin_h2d()
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            meter.end_h2d()

            with autocast_context(cfg.runtime.precision, device.type):
                out = model(
                    input_ids=batch_dev["input_ids"],
                    attention_mask=batch_dev["attention_mask"],
                    labels=batch_dev["labels"],
                    return_mlm_logits=False,
                )
                losses = loss_fn(LossInputs(encoder_out=out, batch=batch_dev))
                loss = losses["total"] / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    fv = float(v.detach())
                    per_step_metrics[f"train/{k}"] = fv
                    # Feed scalar part values back into the composite for
                    # metric-gated warmup / grad_norm_balance.
                    if hasattr(loss_fn, "update_metric"):
                        loss_fn.update_metric(k, fv)
            loss_val += float(loss.detach()) * accum

        # If using GradScaler, unscale before clipping so by-norm/by-value sees
        # the real (unscaled) gradient magnitudes.
        if scaler is not None:
            scaler.unscale_(optim)
        clip_res = clip_gradients(
            list((model.module if isinstance(model, DDP) else model).parameters())
            + [p for p in loss_fn.parameters() if p.requires_grad],
            by_norm=cfg.optim.grad_clip.by_norm,
            by_value=cfg.optim.grad_clip.by_value,
            skip_on_nan=cfg.optim.grad_clip.skip_on_nan,
        )
        per_step_metrics["train/grad_norm"] = clip_res.grad_norm
        per_step_metrics["train/grad_clipped"] = float(clip_res.clipped)
        per_step_metrics["train/grad_skipped"] = float(clip_res.skipped)
        if not clip_res.skipped:
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
        else:
            # Skip optimizer; reset scaler state so it doesn't think the unscale already happened.
            if scaler is not None:
                scaler.update()
        optim.zero_grad(set_to_none=True)
        scheduler.advance()
        if hasattr(loss_fn, "step"):
            loss_fn.step()
        if ema is not None:
            ema.update()

        # Per-step logging cadence
        meter.step()
        per_step_metrics["train/loss_total"] = loss_val
        per_step_metrics["train/lr_mult"] = lr_mult
        per_step_metrics["train/lr"] = cfg.optim.lr * lr_mult
        if (step % max(1, cfg.log.log_every) == 0) or step == total_steps - 1:
            per_step_metrics.update(meter.report())
            ev = TrainEvent(step=step, metrics=per_step_metrics)
            for cb in callbacks:
                cb.on_step_end(ctx, ev)

        # In-loop eval
        if (
            eval_cache is not None
            and cfg.log.eval_every > 0
            and step > 0
            and step % cfg.log.eval_every == 0
        ):
            target = model.module if isinstance(model, DDP) else model
            if ema is not None:
                ema.apply()
            ev_res = None
            try:
                ev_res = run_inloop_eval(
                    target, eval_cache, seq_len=seq_len,
                    log_cfg=cfg.log, device=device, seed=cfg.seed,
                    train_cache=train_cache if cfg.log.enable_probe else None,
                )
            except Exception as exc:  # pragma: no cover — defensive
                # An eval-time OOM or transient failure must NOT bring down a
                # multi-hour training run. Log + continue.
                if is_main:
                    log.warning("in-loop eval failed at step=%d: %s", step, exc)
                # Free any reserved CUDA cache so subsequent training steps
                # don't inherit fragmented allocator state.
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            finally:
                if ema is not None:
                    ema.restore()
            if ev_res is not None:
                ev = TrainEvent(step=step, metrics=ev_res.metrics)
                for cb in callbacks:
                    cb.on_eval_end(ctx, ev)

        # Periodic full-state save (in addition to Checkpointer's safetensors-only saves)
        if is_main and cfg.log.save_every > 0 and step > 0 and step % cfg.log.save_every == 0:
            _save_full_state(
                path=state_path, model=model, optimizer=optim, ema=ema,
                scheduler=scheduler, step=step, rng={},
            )

    # Final save
    if is_main:
        _save_full_state(
            path=state_path, model=model, optimizer=optim, ema=ema,
            scheduler=scheduler, step=total_steps, rng={},
        )

    for cb in callbacks:
        cb.on_train_end(ctx)
    if world > 1:
        dist.destroy_process_group()

    if is_main:
        log.info("training complete: %d steps in %.1fs", total_steps - start_step, time.perf_counter() - train_t0)


__all__ = ["train"]
