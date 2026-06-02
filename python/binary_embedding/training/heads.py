"""Auxiliary heads attached to the encoder body.

The encoder itself stays head-agnostic: it produces `hidden_states` and a
`cls_embedding`. Heads here consume those tensors and return logits suitable
for their loss.

A head is just an `nn.Module` with a known forward signature. Build them
through `build_head(cfg, hidden_size, cls_pool_dim)` so the trainer doesn't
need to know about the constructors.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from binary_embedding.training.config import (
    BYOLLossCfg,
    ClassificationLossCfg,
    ContrastiveLossCfg,
)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Module:
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim, bias=False)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=False), nn.GELU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim, bias=False), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
    return nn.Sequential(*layers)


class ContrastiveHead(nn.Module):
    """SimCSE-style projector. Takes CLS embeddings → L2-normalized projections."""

    def __init__(self, cls_pool_dim: int, cfg: ContrastiveLossCfg) -> None:
        super().__init__()
        self.proj = _mlp(cls_pool_dim, cfg.proj_hidden_dim, cfg.proj_dim, cfg.proj_layers)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        z = self.proj(cls)
        return F.normalize(z, p=2, dim=-1)


class ClassificationHead(nn.Module):
    """Linear classifier on top of CLS embedding."""

    def __init__(self, cls_pool_dim: int, cfg: ClassificationLossCfg) -> None:
        super().__init__()
        self.classifier = nn.Linear(cls_pool_dim, cfg.n_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.classifier(cls)


class BYOLHead(nn.Module):
    """BYOL projector + predictor. The target encoder is held by the loss; this
    is just the projection MLP + the predictor MLP that lives on the online side."""

    def __init__(self, cls_pool_dim: int, cfg: BYOLLossCfg) -> None:
        super().__init__()
        self.projector = _mlp(cls_pool_dim, cfg.proj_hidden_dim, cfg.proj_dim, n_layers=2)
        self.predictor = _mlp(cfg.proj_dim, cfg.proj_hidden_dim, cfg.proj_dim, n_layers=2)

    def forward(self, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proj = self.projector(cls)
        pred = self.predictor(proj)
        return proj, pred


def build_head(loss_cfg: Any, *, hidden_size: int, cls_pool_dim: int) -> nn.Module | None:
    """Return the head module for a single-loss config, or None if no head needed."""
    if isinstance(loss_cfg, ContrastiveLossCfg):
        return ContrastiveHead(cls_pool_dim, loss_cfg)
    if isinstance(loss_cfg, ClassificationLossCfg):
        return ClassificationHead(cls_pool_dim, loss_cfg)
    if isinstance(loss_cfg, BYOLLossCfg):
        return BYOLHead(cls_pool_dim, loss_cfg)
    # MLM uses the encoder's tied head — no extra params.
    return None


__all__ = ["ContrastiveHead", "ClassificationHead", "BYOLHead", "build_head"]
