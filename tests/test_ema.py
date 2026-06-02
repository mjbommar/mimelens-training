"""EMA mechanics: update, apply/restore, state-dict round-trip."""

from __future__ import annotations

import torch
import torch.nn as nn

from binary_embedding.training.regularization import WeightEMA


def _toy() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))


def test_update_moves_shadow_toward_params() -> None:
    model = _toy()
    ema = WeightEMA(model, decay=0.9)
    # Mutate live params.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = next(iter(ema.state.shadows.values())).clone()
    ema.update()
    after = next(iter(ema.state.shadows.values()))
    assert not torch.allclose(before, after)


def test_apply_swaps_in_shadow_then_restore_rolls_back() -> None:
    model = _toy()
    ema = WeightEMA(model, decay=0.5)
    # Mutate live; update so shadow trails live by one step at decay 0.5.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update()
    live_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    ema.apply()
    live_after_apply = {n: p.detach().clone() for n, p in model.named_parameters()}
    # Live should differ from "before" (shadow swapped in).
    assert any(not torch.allclose(live_before[k], live_after_apply[k]) for k in live_before)

    ema.restore()
    live_restored = {n: p.detach().clone() for n, p in model.named_parameters()}
    for k in live_before:
        assert torch.allclose(live_before[k], live_restored[k])


def test_state_dict_roundtrip() -> None:
    a = _toy()
    ema_a = WeightEMA(a, decay=0.9)
    sd = ema_a.state_dict()
    b = _toy()
    ema_b = WeightEMA(b, decay=0.9)
    ema_b.load_state_dict(sd)
    for k in sd:
        assert torch.equal(sd[k], ema_b.state.shadows[k])
