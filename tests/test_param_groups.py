"""Param-group rules: WD=0 for biases/norms, custom rules, LLRD."""

from __future__ import annotations

import pytest
import torch

from binary_embedding.models.encoder import BinaryEncoder, small_encoder_config
from binary_embedding.training.config import OptimCfg, ParamGroupRule, ParamGroupsCfg
from binary_embedding.training.optim import build_param_groups


def _toy_model() -> BinaryEncoder:
    return BinaryEncoder(small_encoder_config(vocab_size=263, max_seq_len=64))


def test_norm_params_get_zero_weight_decay() -> None:
    model = _toy_model()
    groups, glog = build_param_groups(model, OptimCfg(lr=1e-3, weight_decay=0.01))
    no_decay = [g for g in glog if g.weight_decay == 0.0]
    decay = [g for g in glog if g.weight_decay > 0.0]
    assert no_decay and decay
    # All norm.weight tensors must be in the no-decay bucket.
    norm_params = {
        id(p) for n, p in model.named_parameters() if "norm" in n.lower()
    }
    found = {id(p) for g in no_decay for p in g.params}
    assert norm_params.issubset(found)


def test_custom_rule_overrides_group() -> None:
    model = _toy_model()
    cfg = OptimCfg(
        lr=1e-3, weight_decay=0.01,
        param_groups=ParamGroupsCfg(custom_rules=[
            ParamGroupRule(pattern=r"^embed\.", lr_multiplier=0.5, weight_decay=0.0, name="embed_slow"),
        ]),
    )
    _, glog = build_param_groups(model, cfg)
    assert any(g.name == "embed_slow" and g.lr_multiplier == 0.5 for g in glog)


def test_llrd_decays_with_depth() -> None:
    model = _toy_model()
    cfg = OptimCfg(
        lr=1.0, weight_decay=0.01,
        param_groups=ParamGroupsCfg(llrd_decay=0.9),
    )
    _, glog = build_param_groups(model, cfg)
    layer_groups = sorted(
        [g for g in glog if "layer" in g.name],
        key=lambda g: int(g.name.split("layer")[-1]),
    )
    # Last layer (top) has lr_mult=1.0, deeper layers smaller.
    mults = [g.lr_multiplier for g in layer_groups]
    assert mults == sorted(mults), "deeper layers should have smaller lr_multiplier"
    assert mults[-1] == pytest.approx(1.0, abs=1e-6)


def test_all_params_assigned_to_some_group() -> None:
    model = _toy_model()
    _, glog = build_param_groups(model, OptimCfg(lr=1e-3, weight_decay=0.01))
    n_grouped = sum(p.numel() for g in glog for p in g.params)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_grouped == n_total
