"""LR schedule shapes."""

from __future__ import annotations

import pytest

from binary_embedding.training.config import ScheduleCfg
from binary_embedding.training.optim import build_schedule


def _samples(fn, total: int) -> list[float]:
    return [fn(s) for s in (0, total // 4, total // 2, 3 * total // 4, total - 1)]


def test_cosine_warms_then_decays_to_floor() -> None:
    fn = build_schedule(ScheduleCfg(type="cosine", warmup_pct=0.1, min_lr_pct=0.10), total_steps=100)
    s = _samples(fn, 100)
    assert s[0] == 0.0
    assert s[2] > 0.5
    assert s[-1] == pytest.approx(0.10, abs=1e-2)


def test_linear_decays_steadily() -> None:
    fn = build_schedule(ScheduleCfg(type="linear", warmup_pct=0.1, min_lr_pct=0.0), total_steps=100)
    # Warmup first.
    assert fn(0) == 0.0
    assert fn(10) == pytest.approx(1.0, abs=1e-2)
    # Monotone non-increasing across the decay phase.
    decay_samples = [fn(s) for s in range(11, 100)]
    assert all(b <= a + 1e-6 for a, b in zip(decay_samples, decay_samples[1:]))
    assert decay_samples[-1] < 0.05


def test_wsd_has_stable_phase() -> None:
    fn = build_schedule(
        ScheduleCfg(type="wsd", warmup_pct=0.1, stable_pct=0.6, min_lr_pct=0.05),
        total_steps=100,
    )
    # During stable phase (after warmup, before decay) value should be 1.0.
    assert fn(20) == pytest.approx(1.0, abs=1e-6)
    assert fn(50) == pytest.approx(1.0, abs=1e-6)
    assert fn(99) < 0.5


def test_constant_holds_at_one() -> None:
    fn = build_schedule(ScheduleCfg(type="constant", warmup_pct=0.1), total_steps=50)
    assert fn(25) == pytest.approx(1.0, abs=1e-6)
    assert fn(49) == pytest.approx(1.0, abs=1e-6)


def test_warmup_by_bytes() -> None:
    fn = build_schedule(
        ScheduleCfg(type="cosine", warmup_pct=None, warmup_bytes=10_000_000),
        total_steps=100, bytes_per_step=1_000_000,
    )
    # 10M bytes / 1M per step = 10 steps of warmup.
    assert fn(0) == 0.0
    assert fn(5) == pytest.approx(0.5, abs=1e-2)
    assert fn(10) == pytest.approx(1.0, abs=1e-2)


def test_warmup_by_steps() -> None:
    fn = build_schedule(
        ScheduleCfg(type="cosine", warmup_pct=None, warmup_steps=20),
        total_steps=200,
    )
    assert fn(0) == 0.0
    assert fn(20) == pytest.approx(1.0, abs=1e-2)
