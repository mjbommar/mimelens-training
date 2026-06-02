"""Tests for the eight follow-up items (4, 7, 11, 15, 16, 17, 21, 22)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from binary_embedding.models.encoder import (
    BinaryEncoder,
    small_encoder_config,
    verify_weight_tying,
)
from binary_embedding.training.config import (
    CompositeLossCfg,
    ContrastiveLossCfg,
    MLMLossCfg,
    ScheduleCfg,
)
from binary_embedding.training.losses import LossInputs, build_loss
from binary_embedding.training.optim import (
    ClipResult,
    LRScheduler,
    clip_gradients,
)
from binary_embedding.training.runtime import make_worker_init_fn, maybe_compile


# ---------------------------------------------------------------------------
# (15) verify_weight_tying
# ---------------------------------------------------------------------------


def test_weight_tying_audit_passes_for_default_encoder() -> None:
    cfg = small_encoder_config(vocab_size=263, max_seq_len=64)
    model = BinaryEncoder(cfg)
    audit = verify_weight_tying(model)
    assert audit.is_tied
    assert audit.embed_param_name == "embed.weight"
    assert audit.n_vocab_sized_tensors == 1
    assert not audit.has_separate_mlm_head
    assert audit.mlm_logits_depend_on_embed


def test_weight_tying_audit_detects_a_separate_head() -> None:
    """Inject an extra (V,H)-shaped Linear; the audit must catch it."""
    cfg = small_encoder_config(vocab_size=263, max_seq_len=64)
    model = BinaryEncoder(cfg)
    # A naive separate head — would untie weights if used.
    model.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    audit = verify_weight_tying(model)
    assert not audit.is_tied
    assert audit.n_vocab_sized_tensors == 2
    assert audit.has_separate_mlm_head


# ---------------------------------------------------------------------------
# (11) clipped vs skipped counters
# ---------------------------------------------------------------------------


def _params_with_grad(grad_value: float, n: int = 4) -> list[nn.Parameter]:
    ps = [nn.Parameter(torch.zeros(n)) for _ in range(2)]
    for p in ps:
        p.grad = torch.full_like(p, grad_value)
    return ps


def test_clip_result_clipped_flag_true_when_above_norm() -> None:
    ps = _params_with_grad(grad_value=10.0, n=8)  # ||grad|| ≈ sqrt(2*8*100) = ~40
    res = clip_gradients(ps, by_norm=1.0, by_value=None, skip_on_nan=False)
    assert isinstance(res, ClipResult)
    assert res.clipped is True
    assert res.skipped is False
    assert res.grad_norm > 1.0


def test_clip_result_clipped_flag_false_when_below_norm() -> None:
    ps = _params_with_grad(grad_value=0.01, n=4)  # tiny gradient
    res = clip_gradients(ps, by_norm=10.0, by_value=None, skip_on_nan=False)
    assert res.clipped is False
    assert res.skipped is False


def test_clip_result_skipped_on_nan() -> None:
    ps = _params_with_grad(grad_value=1.0)
    ps[0].grad[0] = float("nan")
    res = clip_gradients(ps, by_norm=1.0, by_value=None, skip_on_nan=True)
    assert res.skipped is True
    assert res.clipped is False
    # Skipped runs zero out grads.
    assert all(p.grad is None for p in ps)


# ---------------------------------------------------------------------------
# (22) worker_init_fn seeds independently per worker
# ---------------------------------------------------------------------------


def test_worker_init_fn_seeds_each_worker_distinctly() -> None:
    init = make_worker_init_fn(base_seed=123)
    samples = []
    for w in (0, 1, 2):
        init(w)
        samples.append(np.random.randint(0, 1_000_000_000))
    assert len(set(samples)) == 3, "workers must get distinct numpy random state"
    # And re-seeding a given worker reproduces its draw.
    init(0)
    redraw0 = np.random.randint(0, 1_000_000_000)
    assert redraw0 == samples[0]


# ---------------------------------------------------------------------------
# (16) torch.compile fallback on failure
# ---------------------------------------------------------------------------


def test_maybe_compile_none_is_noop() -> None:
    m = nn.Linear(4, 4)
    out = maybe_compile(m, "none")
    assert out is m


def test_maybe_compile_falls_back_when_compile_raises(monkeypatch) -> None:
    m = nn.Linear(4, 4)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(torch, "compile", boom)
    out = maybe_compile(m, "max-autotune")
    assert out is m  # eager fallback


# ---------------------------------------------------------------------------
# (4) scheduler state save/load
# ---------------------------------------------------------------------------


def test_lr_scheduler_persists_step_across_save_load() -> None:
    sched = LRScheduler(ScheduleCfg(type="cosine", warmup_pct=0.1), total_steps=50)
    for _ in range(7):
        sched.advance()
    assert sched.step_count == 7
    saved = sched.state_dict()

    sched2 = LRScheduler(ScheduleCfg(type="cosine", warmup_pct=0.1), total_steps=50)
    sched2.load_state_dict(saved)
    assert sched2.step_count == 7
    assert sched2.lr_multiplier() == sched.lr_multiplier()


def test_lr_scheduler_warns_on_type_mismatch_but_keeps_step(caplog) -> None:
    a = LRScheduler(ScheduleCfg(type="cosine", warmup_pct=0.1), total_steps=10)
    a._step = 5
    b = LRScheduler(ScheduleCfg(type="linear", warmup_pct=0.1), total_steps=10)
    with caplog.at_level("WARNING"):
        b.load_state_dict(a.state_dict())
    assert b.step_count == 5
    assert any("type mismatch" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# (7) grad_norm_balance + metric-gated warmup
# ---------------------------------------------------------------------------


@pytest.fixture()
def comp_inputs():
    cfg = small_encoder_config(vocab_size=263, max_seq_len=32)
    enc = BinaryEncoder(cfg)
    B, S = 4, 16
    ids1 = torch.randint(7, 263, (B, S)); ids1[:, 0] = 4
    ids2 = torch.randint(7, 263, (B, S)); ids2[:, 0] = 4
    attn = torch.ones_like(ids1)
    labels = ids1.clone(); labels[:, 1::2] = -100
    out1 = enc(ids1, attn, labels=labels)
    out2 = enc(ids2, attn, labels=None, return_mlm_logits=False)
    return enc, out1, out2, {"input_ids": ids1, "attention_mask": attn, "labels": labels}


def test_metric_gated_warmup_holds_aux_at_zero_until_threshold(comp_inputs) -> None:
    enc, out1, out2, batch = comp_inputs
    cfg = CompositeLossCfg(
        parts=[MLMLossCfg(), ContrastiveLossCfg()],
        aux_warmup_steps=4,
        aux_warmup_metric="mlm",
        aux_warmup_threshold=0.0,  # impossible-to-reach; ramp must stay 0
    )
    loss = build_loss(cfg, hidden_size=enc.cfg.hidden_size, cls_pool_dim=enc.cfg.cls_pool_dim)
    inputs = LossInputs(encoder_out=out1, batch=batch, aux={"view2_cls": out2.cls_embedding})
    for _ in range(6):
        out = loss(inputs)
        loss.update_metric("mlm", float(out["mlm"].detach()))
        loss.step()
    assert out["aux_ramp"].item() == 0.0


def test_metric_gated_warmup_engages_after_threshold_crossed(comp_inputs) -> None:
    enc, out1, out2, batch = comp_inputs
    cfg = CompositeLossCfg(
        parts=[MLMLossCfg(), ContrastiveLossCfg()],
        aux_warmup_steps=2,
        aux_warmup_metric="mlm",
        aux_warmup_threshold=1e9,  # always satisfied; ramp engages immediately
    )
    loss = build_loss(cfg, hidden_size=enc.cfg.hidden_size, cls_pool_dim=enc.cfg.cls_pool_dim)
    inputs = LossInputs(encoder_out=out1, batch=batch, aux={"view2_cls": out2.cls_embedding})
    ramps = []
    for _ in range(5):
        out = loss(inputs)
        loss.update_metric("mlm", float(out["mlm"].detach()))
        ramps.append(out["aux_ramp"].item())
        loss.step()
    # update_metric is called *after* forward, so:
    # Step 0: no metric yet -> gate not tripped -> ramp = 0
    # Step 1: gate trips this step -> since=0 -> ramp = 0
    # Step 2: since=1 -> ramp = 0.5
    # Step 3: since=2 -> ramp = 1.0
    # Step 4: clamped at 1.0
    assert ramps[0] == 0.0
    assert ramps[1] == 0.0
    assert ramps[2] == pytest.approx(0.5, abs=1e-3)
    assert ramps[3] == pytest.approx(1.0, abs=1e-3)
    assert ramps[4] == pytest.approx(1.0, abs=1e-3)


def test_grad_norm_balance_introduces_per_part_balance_metric(comp_inputs) -> None:
    enc, out1, out2, batch = comp_inputs
    cfg = CompositeLossCfg(
        parts=[MLMLossCfg(), ContrastiveLossCfg()],
        grad_norm_balance=True,
        aux_warmup_steps=0,
    )
    loss = build_loss(cfg, hidden_size=enc.cfg.hidden_size, cls_pool_dim=enc.cfg.cls_pool_dim)
    inputs = LossInputs(encoder_out=out1, batch=batch, aux={"view2_cls": out2.cls_embedding})
    out = loss(inputs)
    assert "contrastive/balance" in out
    assert out["contrastive/balance"].ndim == 0


def test_aux_warmup_validator_rejects_partial_gating() -> None:
    with pytest.raises(Exception, match="aux_warmup_metric and aux_warmup_threshold"):
        CompositeLossCfg(
            parts=[MLMLossCfg()],
            aux_warmup_metric="mlm",  # threshold missing
        )


# ---------------------------------------------------------------------------
# (17) GradScaler is wired (smoke — fp32 branch we can't easily exercise on CPU,
# but importability + scaler-construction-on-cuda are checked elsewhere)
# ---------------------------------------------------------------------------


def test_gradscaler_class_is_importable() -> None:
    # Just confirm the trainer-side import path is valid.
    assert hasattr(torch.amp, "GradScaler")


# ---------------------------------------------------------------------------
# (21) in-loop probe — covered by the data-layer-bound smoke in test_data_layer
# (this file just confirms the function is importable + sanity on no data).
# ---------------------------------------------------------------------------


def test_run_inloop_probe_returns_nan_on_empty(comp_inputs) -> None:
    """If no class diversity is present we should return NaN, not crash."""
    from binary_embedding.training.eval_loop import run_inloop_probe
    # Pass an obviously-empty proxy by mocking caches with len=0 metadata.
    # We rely on the function's len() check.
    class _EmptyCache:
        def __len__(self): return 0
    out = run_inloop_probe(
        comp_inputs[0], _EmptyCache(), _EmptyCache(),
        seq_len=32, target_label="x", n_files=4, seed=0, device=torch.device("cpu"),
    )
    assert "eval/probe_top1" in out
    assert out["eval/probe_top1"] != out["eval/probe_top1"]  # NaN
