"""Loss objects + composite + aux ramping + grad coverage."""

from __future__ import annotations

import pytest
import torch

from binary_embedding.models.encoder import BinaryEncoder, small_encoder_config
from binary_embedding.training.config import (
    BYOLLossCfg,
    ClassificationLossCfg,
    CompositeLossCfg,
    ContrastiveLossCfg,
    MLMLossCfg,
)
from binary_embedding.training.losses import LossInputs, build_loss


@pytest.fixture()
def encoder():
    return BinaryEncoder(small_encoder_config(vocab_size=263, max_seq_len=32))


@pytest.fixture()
def two_views(encoder):
    B, S = 4, 16
    ids1 = torch.randint(7, 263, (B, S)); ids1[:, 0] = 4
    ids2 = torch.randint(7, 263, (B, S)); ids2[:, 0] = 4
    attn = torch.ones_like(ids1)
    labels = ids1.clone(); labels[:, 1::2] = -100
    out1 = encoder(ids1, attn, labels=labels)
    out2 = encoder(ids2, attn, labels=None, return_mlm_logits=False)
    return out1, out2, {"input_ids": ids1, "attention_mask": attn, "labels": labels}


def test_mlm_loss_returns_scalar(encoder, two_views) -> None:
    out1, _, batch = two_views
    loss = build_loss(MLMLossCfg(weight=1.0), hidden_size=encoder.cfg.hidden_size, cls_pool_dim=encoder.cfg.cls_pool_dim)
    o = loss(LossInputs(encoder_out=out1, batch=batch))
    assert "mlm" in o and "total" in o
    assert o["total"].ndim == 0


def test_contrastive_uses_view2(encoder, two_views) -> None:
    out1, out2, batch = two_views
    loss = build_loss(ContrastiveLossCfg(temperature=0.05), hidden_size=encoder.cfg.hidden_size, cls_pool_dim=encoder.cfg.cls_pool_dim)
    o = loss(LossInputs(encoder_out=out1, batch=batch, aux={"view2_cls": out2.cls_embedding}))
    assert o["total"].item() > 0.0


def test_contrastive_no_view2_is_noop(encoder, two_views) -> None:
    out1, _, batch = two_views
    loss = build_loss(ContrastiveLossCfg(), hidden_size=encoder.cfg.hidden_size, cls_pool_dim=encoder.cfg.cls_pool_dim)
    o = loss(LossInputs(encoder_out=out1, batch=batch))
    assert o["total"].item() == 0.0


def test_classification_uses_targets(encoder, two_views) -> None:
    out1, _, batch = two_views
    loss = build_loss(ClassificationLossCfg(n_classes=4, weight=0.2), hidden_size=encoder.cfg.hidden_size, cls_pool_dim=encoder.cfg.cls_pool_dim)
    targets = torch.tensor([0, 1, 2, 3])
    o = loss(LossInputs(encoder_out=out1, batch=batch, aux={"target_class": targets}))
    assert o["total"].item() > 0.0


def test_byol_target_decay(encoder, two_views) -> None:
    out1, out2, batch = two_views
    loss = build_loss(BYOLLossCfg(target_decay=0.99), hidden_size=encoder.cfg.hidden_size, cls_pool_dim=encoder.cfg.cls_pool_dim)
    online_before = next(loss.online_head.parameters()).detach().clone()
    target_before = next(loss.target_head.parameters()).detach().clone()
    assert torch.allclose(online_before, target_before)
    # Mutate online → update_target should not equal online (close, but different).
    next(loss.online_head.parameters()).data.add_(0.5)
    loss.update_target()
    after = next(loss.target_head.parameters()).detach().clone()
    assert not torch.allclose(after, online_before)
    assert not torch.allclose(after, next(loss.online_head.parameters()).detach())


def test_composite_combines_with_aux_warmup(encoder, two_views) -> None:
    out1, out2, batch = two_views
    cfg = CompositeLossCfg(parts=[MLMLossCfg(), ContrastiveLossCfg()], aux_warmup_steps=4)
    loss = build_loss(cfg, hidden_size=encoder.cfg.hidden_size, cls_pool_dim=encoder.cfg.cls_pool_dim)
    inputs = LossInputs(encoder_out=out1, batch=batch, aux={"view2_cls": out2.cls_embedding})

    samples = []
    for s in range(5):
        o = loss(inputs)
        samples.append((o["aux_ramp"].item(), o["total"].item(), o["mlm"].item(), o["contrastive"].item()))
        loss.step()
    ramps = [r for r, _, _, _ in samples]
    assert ramps == [0.0, 0.25, 0.5, 0.75, 1.0]
    # At step 0 (ramp=0) total ≈ mlm; at step 4 (ramp=1) total ≈ mlm + contrastive*weight
    r0 = samples[0]; r4 = samples[4]
    assert abs(r0[1] - r0[2]) < 1e-3
    assert r4[1] >= r4[2]  # contrastive contributes at the end


def test_composite_backward_populates_grads(encoder, two_views) -> None:
    out1, out2, batch = two_views
    cfg = CompositeLossCfg(parts=[MLMLossCfg(), ContrastiveLossCfg()])
    loss = build_loss(cfg, hidden_size=encoder.cfg.hidden_size, cls_pool_dim=encoder.cfg.cls_pool_dim)
    o = loss(LossInputs(encoder_out=out1, batch=batch, aux={"view2_cls": out2.cls_embedding}))
    o["total"].backward()
    n = sum(1 for p in list(encoder.parameters()) + list(loss.parameters())
            if p.grad is not None and p.grad.abs().sum() > 0)
    assert n > 5
