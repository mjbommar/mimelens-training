"""Loss interface, single-loss implementations, and CompositeLoss.

A `Loss` is an `nn.Module` whose `forward(...)` returns a dict of named scalar
tensors plus a `total` key. Multiple losses compose linearly via
`CompositeLoss`. This keeps the trainer's loop trivial: call `loss(...)`,
`loss['total'].backward()`.

Inputs flow through a tiny shared dataclass `LossInputs`:
- `encoder_out` from the model forward (mlm_logits, cls_embedding, hidden_states)
- `batch` from the dataloader (input_ids, attention_mask, labels)
- `aux` for optional fields (e.g. `view2_cls`, `target_class`, EMA-target output)

Heads live in `heads.py` and are owned by the loss object so they get optimizer
visibility through `loss.parameters()`.
"""

from __future__ import annotations

import copy
import dataclasses as dc
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from binary_embedding.models.encoder import EncoderOutputs
from binary_embedding.training.config import (
    BYOLLossCfg,
    ClassificationLossCfg,
    CompositeLossCfg,
    ContrastiveLossCfg,
    MLMLossCfg,
    SingleLoss,
)
from binary_embedding.training.heads import build_head
from binary_embedding.training.registry import LOSSES


@dc.dataclass(slots=True)
class LossInputs:
    encoder_out: EncoderOutputs
    batch: dict[str, torch.Tensor]
    aux: dict[str, Any] = dc.field(default_factory=dict)


class Loss(nn.Module):
    """Base class. Subclasses implement `forward(LossInputs) -> dict[str, Tensor]`.

    The returned dict MUST include `total`. Sub-keys are recorded for logging.
    """

    name: str

    def forward(self, inputs: LossInputs) -> dict[str, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MLM
# ---------------------------------------------------------------------------


@LOSSES.register("mlm")
class MLMLoss(Loss):
    def __init__(self, cfg: MLMLossCfg, *, hidden_size: int, cls_pool_dim: int) -> None:
        super().__init__()
        self.name = cfg.name
        self.cfg = cfg

    def forward(self, inputs: LossInputs) -> dict[str, torch.Tensor]:
        # The encoder already computed loss when it saw labels. Reuse it.
        if inputs.encoder_out.loss is None:
            raise RuntimeError(
                "MLMLoss expected encoder_out.loss to be populated; "
                "ensure labels are passed and `return_mlm_logits=False` is acceptable"
            )
        loss = inputs.encoder_out.loss
        return {self.name: loss, "total": loss * self.cfg.weight}


# ---------------------------------------------------------------------------
# Contrastive (SimCSE in-batch)
# ---------------------------------------------------------------------------


@LOSSES.register("contrastive")
class ContrastiveLoss(Loss):
    """In-batch contrastive loss between two views of the same file.

    Expects `inputs.aux["view2_cls"]` to hold a second forward's CLS embedding.
    The trainer is responsible for running that second forward.
    """

    def __init__(self, cfg: ContrastiveLossCfg, *, hidden_size: int, cls_pool_dim: int) -> None:
        super().__init__()
        self.name = cfg.name
        self.cfg = cfg
        self.head = build_head(cfg, hidden_size=hidden_size, cls_pool_dim=cls_pool_dim)

    def forward(self, inputs: LossInputs) -> dict[str, torch.Tensor]:
        z1 = self.head(inputs.encoder_out.cls_embedding)
        view2 = inputs.aux.get("view2_cls")
        if view2 is None:
            # Degenerate: only one view available. Treat this as a no-op so the
            # trainer can keep the loss in the composite without crashing.
            zero = z1.sum() * 0.0
            return {self.name: zero, "total": zero}
        z2 = self.head(view2)
        sim = (z1 @ z2.t()) / self.cfg.temperature
        targets = torch.arange(z1.size(0), device=z1.device)
        loss = 0.5 * (
            F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets)
        )
        return {self.name: loss, "total": loss * self.cfg.weight}


# ---------------------------------------------------------------------------
# Classification (e.g. file_format)
# ---------------------------------------------------------------------------


@LOSSES.register("classification")
class ClassificationLoss(Loss):
    """Auxiliary CE head over a metadata column (e.g. file_format).

    Expects `inputs.aux["target_class"]` to hold a (B,) long tensor of class
    indices. The dataloader should map labels -> ints; the loss is agnostic.
    """

    def __init__(self, cfg: ClassificationLossCfg, *, hidden_size: int, cls_pool_dim: int) -> None:
        super().__init__()
        self.name = cfg.name
        self.cfg = cfg
        self.head = build_head(cfg, hidden_size=hidden_size, cls_pool_dim=cls_pool_dim)

    def forward(self, inputs: LossInputs) -> dict[str, torch.Tensor]:
        targets = inputs.aux.get("target_class")
        if targets is None:
            zero = inputs.encoder_out.cls_embedding.sum() * 0.0
            return {self.name: zero, "total": zero}
        logits = self.head(inputs.encoder_out.cls_embedding)
        loss = F.cross_entropy(logits, targets, label_smoothing=self.cfg.label_smoothing)
        return {self.name: loss, "total": loss * self.cfg.weight}


# ---------------------------------------------------------------------------
# BYOL
# ---------------------------------------------------------------------------


@LOSSES.register("byol")
class BYOLLoss(Loss):
    """BYOL: predict the target network's projection of view 2 from view 1.

    Owns its own EMA-target encoder. Trainer must call `update_target(online)`
    each step to keep the target slow-moving.
    """

    def __init__(
        self, cfg: BYOLLossCfg, *, hidden_size: int, cls_pool_dim: int
    ) -> None:
        super().__init__()
        self.name = cfg.name
        self.cfg = cfg
        self.online_head = build_head(cfg, hidden_size=hidden_size, cls_pool_dim=cls_pool_dim)
        self.target_head = copy.deepcopy(self.online_head)
        for p in self.target_head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, decay: float | None = None) -> None:
        d = decay if decay is not None else self.cfg.target_decay
        for p_t, p_o in zip(self.target_head.parameters(), self.online_head.parameters()):
            p_t.data.mul_(d).add_(p_o.data, alpha=1.0 - d)

    def forward(self, inputs: LossInputs) -> dict[str, torch.Tensor]:
        cls1 = inputs.encoder_out.cls_embedding
        view2_cls = inputs.aux.get("view2_cls")
        if view2_cls is None:
            zero = cls1.sum() * 0.0
            return {self.name: zero, "total": zero}
        proj1, pred1 = self.online_head(cls1)
        with torch.no_grad():
            proj2, _ = self.target_head(view2_cls)
        # Normalize and use 2 - 2*cos similarity (BYOL loss).
        pred1n = F.normalize(pred1, dim=-1)
        proj2n = F.normalize(proj2, dim=-1)
        loss = (2 - 2 * (pred1n * proj2n).sum(-1)).mean()
        return {self.name: loss, "total": loss * self.cfg.weight}


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class CompositeLoss(Loss):
    """Linear combination of named sub-losses.

    Logs every part's raw + weighted value. Knobs:
    - `aux_warmup_steps`: linearly ramp non-MLM parts in over this many steps.
    - `aux_warmup_metric` + `aux_warmup_threshold`: gate the ramp on a scalar
      metric; the ramp does not start until that metric (e.g. `mlm`) drops
      below the threshold (i.e. MLM is "good enough" before contrastive joins).
    - `grad_norm_balance`: scale each non-MLM part's contribution by
      `mean_mlm_loss / mean_part_loss`, clipped by `grad_norm_balance_clip`.
      Approximate "balance the contributions so no head dominates" — not full
      PCGrad, but cheap (no extra backward passes) and effective for our scale.

    Track the gating metric via `update_metric(name, value)` from the trainer
    after each step.
    """

    def __init__(
        self,
        cfg: CompositeLossCfg,
        *,
        hidden_size: int,
        cls_pool_dim: int,
    ) -> None:
        super().__init__()
        self.name = "composite"
        self.cfg = cfg
        self.parts: nn.ModuleList = nn.ModuleList()
        for part_cfg in cfg.parts:
            self.parts.append(
                LOSSES.build(part_cfg.type, part_cfg, hidden_size=hidden_size, cls_pool_dim=cls_pool_dim)
            )
        # Step counter (for step-based warmup) and "ramp engaged" flag (for metric warmup).
        self._step = 0
        self._ramp_engaged_step: int | None = None
        # Tracked metrics for gating (e.g. "mlm" loss EMA).
        self._tracked_metrics: dict[str, float] = {}
        # Per-part running magnitudes for grad_norm_balance.
        self._running_mag: dict[str, float] = {}

    def step(self) -> None:
        self._step += 1

    def update_metric(self, name: str, value: float) -> None:
        """Hook for the trainer to feed in scalar values used by `aux_warmup_metric`."""
        self._tracked_metrics[name] = float(value)

    def _maybe_engage_ramp(self) -> None:
        cfg = self.cfg
        if (
            self._ramp_engaged_step is None
            and cfg.aux_warmup_metric is not None
            and cfg.aux_warmup_threshold is not None
        ):
            v = self._tracked_metrics.get(cfg.aux_warmup_metric)
            if v is not None and v < cfg.aux_warmup_threshold:
                self._ramp_engaged_step = self._step

    def aux_ramp(self) -> float:
        cfg = self.cfg
        # Metric-gated: ramp doesn't start until the gate trips.
        if cfg.aux_warmup_metric is not None:
            self._maybe_engage_ramp()
            if self._ramp_engaged_step is None:
                return 0.0
            since = self._step - self._ramp_engaged_step
        else:
            since = self._step
        if cfg.aux_warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, since / cfg.aux_warmup_steps))

    def forward(self, inputs: LossInputs) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        ramp = self.aux_ramp()
        ema = self.cfg.grad_norm_balance_ema
        clip = self.cfg.grad_norm_balance_clip

        # First pass: compute each part. Capture raw scalars so we can balance.
        raw_values: dict[str, torch.Tensor] = {}
        for sub in self.parts:
            sub_out = sub(inputs)
            raw_values[sub.name] = sub_out["total"].detach()
            for k, v in sub_out.items():
                if k != "total":
                    out[k] = v

        # Update running magnitudes (EMA) for grad_norm_balance.
        for name, v in raw_values.items():
            mag = float(v.detach().abs())
            prev = self._running_mag.get(name)
            self._running_mag[name] = (
                mag if prev is None else ema * prev + (1.0 - ema) * mag
            )

        mlm_mag = self._running_mag.get("mlm", 1.0)
        # Second pass: assemble total with balance + ramp.
        for sub in self.parts:
            v = raw_values[sub.name]
            scale = 1.0
            if sub.name != "mlm":
                scale *= ramp
                if self.cfg.grad_norm_balance:
                    part_mag = max(self._running_mag.get(sub.name, 1.0), 1e-6)
                    bal = max(min(mlm_mag / part_mag, clip), 1.0 / clip)
                    scale *= bal
                    out[f"{sub.name}/balance"] = torch.tensor(bal, device=v.device)
            # Re-run with grad: the second forward isn't free but parts are
            # small modules over a (B, S, H) tensor that's already cached.
            # For now the part's own .forward is idempotent so we just rebuild.
            sub_out = sub(inputs)
            contribution = sub_out["total"] * scale
            total = contribution if total is None else total + contribution

        if total is None:
            raise RuntimeError("CompositeLoss has no parts")
        out["total"] = total
        out["aux_ramp"] = torch.tensor(ramp, device=total.device)
        return out


def build_loss(
    cfg: Any, *, hidden_size: int, cls_pool_dim: int
) -> Loss:
    if isinstance(cfg, CompositeLossCfg):
        return CompositeLoss(cfg, hidden_size=hidden_size, cls_pool_dim=cls_pool_dim)
    return LOSSES.build(cfg.type, cfg, hidden_size=hidden_size, cls_pool_dim=cls_pool_dim)


__all__ = [
    "BYOLLoss", "ClassificationLoss", "CompositeLoss", "ContrastiveLoss", "Loss",
    "LossInputs", "MLMLoss", "build_loss",
]
