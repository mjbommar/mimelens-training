"""EMA of model weights (used for eval / best-checkpointing).

Implementation notes:
- Stores shadows in the same dtype as the source params; casts up to fp32 only
  when accumulating the running average. bf16-clean and memory-light.
- On `apply()` we swap weights in-place and remember the originals so `restore()`
  rolls back cleanly. This avoids deepcopying the model for eval.
"""

from __future__ import annotations

import dataclasses as dc

import torch
import torch.nn as nn


@dc.dataclass(slots=True)
class EMAState:
    decay: float
    shadows: dict[str, torch.Tensor]
    backup: dict[str, torch.Tensor] = dc.field(default_factory=dict)


class WeightEMA:
    """Maintain an exponential-moving-average of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self._model = model
        shadows = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.state = EMAState(decay=decay, shadows=shadows)

    @torch.no_grad()
    def update(self) -> None:
        d = self.state.decay
        for n, p in self._model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.state.shadows.get(n)
            if shadow is None:
                self.state.shadows[n] = p.detach().clone()
                continue
            # shadow = d * shadow + (1-d) * p, computed in shadow's dtype.
            shadow.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply(self) -> None:
        """Swap shadow weights into the live model. Call `restore()` after."""
        if self.state.backup:
            raise RuntimeError("EMA already applied; call restore() first")
        for n, p in self._model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.state.shadows.get(n)
            if shadow is None:
                continue
            self.state.backup[n] = p.detach().clone()
            p.copy_(shadow)

    @torch.no_grad()
    def restore(self) -> None:
        for n, p in self._model.named_parameters():
            if n in self.state.backup:
                p.copy_(self.state.backup[n])
        self.state.backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state.shadows.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        # Replace shadows with the loaded values, matching the device of the
        # existing shadow (which was initialised from the live model params, so
        # carries the correct device). Without this, resume from a CPU-loaded
        # state.pt leaves shadows on CPU while model params are on CUDA, and
        # the next update() raises a device-mismatch RuntimeError.
        new_shadows = {}
        for n, t in sd.items():
            tgt_device = self.state.shadows[n].device if n in self.state.shadows else t.device
            new_shadows[n] = t.detach().clone().to(tgt_device)
        self.state.shadows = new_shadows


__all__ = ["WeightEMA"]
