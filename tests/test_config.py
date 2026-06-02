"""Tests for the pydantic v2 config + CLI overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from binary_embedding.training.config import (
    ClassificationLossCfg,
    CompositeLossCfg,
    ContrastiveLossCfg,
    MLMLossCfg,
    RunConfig,
    apply_overrides,
    dump_resolved_config,
    load_config,
)


def test_minimal_config_validates() -> None:
    cfg = RunConfig(run_id="x", schedule_budget={"target_steps": 100})
    assert cfg.optim.lr == 5e-4
    assert cfg.loss.type == "mlm"


def test_invalid_mask_ratio_rejected() -> None:
    with pytest.raises(Exception):
        RunConfig(run_id="x", schedule_budget={"target_steps": 100}, data={"mask_ratio": 1.5})


def test_extra_keys_forbidden() -> None:
    with pytest.raises(Exception):
        RunConfig.model_validate({"run_id": "x", "schedule_budget": {"target_steps": 1}, "wibble": 7})


def test_dual_warmup_rejected() -> None:
    with pytest.raises(Exception, match="exactly one of warmup"):
        RunConfig.model_validate({
            "run_id": "x",
            "schedule_budget": {"target_steps": 1},
            "schedule": {"warmup_pct": 0.1, "warmup_steps": 100},
        })


def test_dual_target_rejected() -> None:
    with pytest.raises(Exception, match="exactly one of target"):
        RunConfig.model_validate({
            "run_id": "x",
            "schedule_budget": {"target_bytes": 1, "target_steps": 1},
        })


def test_no_target_rejected() -> None:
    with pytest.raises(Exception, match="exactly one of target"):
        RunConfig.model_validate({"run_id": "x"})


def test_loss_discriminated_union_mlm() -> None:
    cfg = RunConfig.model_validate({
        "run_id": "x", "schedule_budget": {"target_steps": 1},
        "loss": {"type": "mlm", "weight": 0.7},
    })
    assert isinstance(cfg.loss, MLMLossCfg)
    assert cfg.loss.weight == 0.7


def test_loss_discriminated_union_composite() -> None:
    cfg = RunConfig.model_validate({
        "run_id": "x", "schedule_budget": {"target_steps": 1},
        "loss": {
            "type": "composite",
            "parts": [{"type": "mlm"}, {"type": "contrastive", "temperature": 0.07}],
            "aux_warmup_steps": 100,
        },
    })
    assert isinstance(cfg.loss, CompositeLossCfg)
    assert {p.type for p in cfg.loss.parts} == {"mlm", "contrastive"}
    contr = next(p for p in cfg.loss.parts if isinstance(p, ContrastiveLossCfg))
    assert contr.temperature == 0.07


def test_classification_loss_args_pass_through() -> None:
    cfg = RunConfig.model_validate({
        "run_id": "x", "schedule_budget": {"target_steps": 1},
        "loss": {"type": "classification", "n_classes": 8, "label_smoothing": 0.1},
    })
    assert isinstance(cfg.loss, ClassificationLossCfg)
    assert cfg.loss.n_classes == 8


def test_apply_overrides_typed() -> None:
    raw = {"run_id": "x", "schedule_budget": {"target_steps": 1}}
    raw2 = apply_overrides(raw, [
        "optim.lr=3e-4",
        "schedule_budget.target_bytes=5_000_000",
        "schedule_budget.target_steps=null",
        "reg.ema_decay=0.999",
        "data.doc_sampling=sqrt_bytes",
    ])
    cfg = RunConfig.model_validate(raw2)
    assert cfg.optim.lr == 3e-4
    assert cfg.schedule_budget.target_bytes == 5_000_000
    assert cfg.schedule_budget.target_steps is None
    assert cfg.reg.ema_decay == 0.999
    assert cfg.data.doc_sampling == "sqrt_bytes"


def test_load_yaml_with_overrides(tmp_path: Path) -> None:
    yaml_path = tmp_path / "smoke.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "run_id": "y", "schedule_budget": {"target_steps": 100},
    }))
    cfg = load_config(yaml_path, overrides=["optim.lr=7e-5"])
    assert cfg.optim.lr == 7e-5
    assert cfg.run_id == "y"


def test_dump_resolved_yaml_round_trips(tmp_path: Path) -> None:
    cfg = RunConfig(run_id="z", schedule_budget={"target_steps": 10})
    out = dump_resolved_config(cfg, tmp_path)
    text = out.read_text()
    assert "run_id: z" in text
    assert "config_hash:" in text


def test_config_hash_is_stable() -> None:
    a = RunConfig(run_id="a", schedule_budget={"target_steps": 1}).hash_short()
    b = RunConfig(run_id="a", schedule_budget={"target_steps": 1}).hash_short()
    c = RunConfig(run_id="a", schedule_budget={"target_steps": 2}).hash_short()
    assert a == b
    assert a != c
