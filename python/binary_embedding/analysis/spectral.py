"""Layer / spectral diagnostics for trained encoders.

Per-layer reports:
- Weight singular values for the linear projections (Q/K/V/O of attention,
  W_gate / W_up / W_down of GeGLU). From these we report:
    * spectral_norm = sigma_1
    * cond_number   = sigma_1 / sigma_min (truncated to non-zero singular values)
    * effective_rank ≈ exp(H(p)) where p = sigma_i / sum(sigma_i)
    * stable_rank   = ||W||_F^2 / sigma_1^2
- Activation effective rank — PCA spectrum of the (B*S, H) hidden states at
  each layer for a tiny held-out batch.
- Attention entropy per head per layer — mean Shannon entropy of softmax(QK/√d)
  over a held-out batch. High = uniform attention; low = concentrated.

The report is shape-stable across variants so cross-variant comparison is just
a parquet join.
"""

from __future__ import annotations

import dataclasses as dc
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from binary_embedding.constants import CLS_ID, PAD_ID, SEP_ID
from binary_embedding.data.dataset import TokenCache
from binary_embedding.models.encoder import BinaryEncoder


# ---------------------------------------------------------------------------
# Weight-only diagnostics (no inputs needed)
# ---------------------------------------------------------------------------


@dc.dataclass(slots=True)
class WeightSpectrum:
    name: str
    shape: tuple[int, int]
    spectral_norm: float
    cond_number: float
    effective_rank: float
    stable_rank: float
    frobenius_norm: float


def _spectrum(weight: torch.Tensor) -> tuple[np.ndarray, float]:
    """Return singular values + Frobenius norm."""
    w = weight.detach().float().cpu()
    if w.ndim != 2:
        w = w.reshape(w.shape[0], -1)
    s = torch.linalg.svdvals(w).numpy()
    fro = float(torch.linalg.norm(w).item())
    return s, fro


def _effective_rank(s: np.ndarray) -> float:
    """exp(H(p)) where p_i = s_i / sum(s_i). Counts a "soft" number of dimensions."""
    p = s / max(float(s.sum()), 1e-12)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return math.exp(h)


def _stable_rank(s: np.ndarray, fro: float) -> float:
    if s.size == 0 or s[0] <= 0:
        return 0.0
    return (fro ** 2) / (float(s[0]) ** 2)


def _cond_number(s: np.ndarray, eps: float = 1e-12) -> float:
    nz = s[s > eps]
    if nz.size < 2:
        return float("nan")
    return float(nz[0] / nz[-1])


def weight_spectra(model: BinaryEncoder) -> list[WeightSpectrum]:
    """Walk every (layer, sub-projection) and emit a `WeightSpectrum`.

    Module names are stable (`layers.<i>.attn.qkv`, `attn.out`, `ffn.w_gate`,
    `ffn.w_up`, `ffn.w_down`), so cross-variant joins are trivial.
    """
    out: list[WeightSpectrum] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Restrict to encoder backbone projections + cls pool.
        if not (name.startswith("layers.") or name == "cls_pool"):
            continue
        s, fro = _spectrum(module.weight)
        out.append(
            WeightSpectrum(
                name=name,
                shape=tuple(module.weight.shape),
                spectral_norm=float(s[0]) if s.size else 0.0,
                cond_number=_cond_number(s),
                effective_rank=_effective_rank(s),
                stable_rank=_stable_rank(s, fro),
                frobenius_norm=fro,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Activation diagnostics (one tiny forward pass)
# ---------------------------------------------------------------------------


@dc.dataclass(slots=True)
class ActivationSpectrum:
    layer: int
    effective_rank: float       # over (B*S_real, H) PCA
    activation_norm_mean: float
    activation_norm_std: float


@dc.dataclass(slots=True)
class AttentionEntropy:
    layer: int
    head: int
    entropy_mean: float          # mean Shannon entropy of attention softmax (nats)
    entropy_p10: float
    entropy_p90: float


def _build_probe_batch(cache: TokenCache, n_files: int, seq_len: int, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    n = len(cache)
    if n == 0:
        raise ValueError("cache is empty; cannot build probe batch")
    fids = rng.choice(n, size=min(n_files, n), replace=False).tolist()
    body_budget = seq_len - 2
    rows_ids: list[list[int]] = []
    rows_attn: list[list[int]] = []
    for fi in fids:
        body = cache.file_window(fi).tolist()[:body_budget]
        ids = [CLS_ID, *body, SEP_ID]
        pad = seq_len - len(ids)
        ids += [PAD_ID] * pad
        attn = [1] * (len(body) + 2) + [0] * pad
        rows_ids.append(ids); rows_attn.append(attn)
    return (
        torch.tensor(rows_ids, dtype=torch.long, device=device),
        torch.tensor(rows_attn, dtype=torch.long, device=device),
    )


@torch.no_grad()
def activation_and_attention_diagnostics(
    model: BinaryEncoder,
    cache: TokenCache,
    *,
    n_files: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> tuple[list[ActivationSpectrum], list[AttentionEntropy]]:
    """Forward a small batch with hooks; report activation rank + attn entropy."""
    was_training = model.training
    model.eval()

    # Capture per-layer hidden states via hooks on each EncoderBlock output.
    layer_outs: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_layer_hook(idx: int):
        def _hook(_mod, _inp, out):
            layer_outs[idx] = out.detach()
        return _hook

    for i, blk in enumerate(model.layers):
        hooks.append(blk.register_forward_hook(_make_layer_hook(i)))

    # Capture attention scores by intercepting the QKV projection inside each
    # `MultiHeadSelfAttention`. We mimic the forward but recompute the softmax
    # explicitly so we can read it.
    attn_entropies: list[AttentionEntropy] = []

    def _attn_hook(idx: int):
        def _hook(mod, inputs, _output):
            x, attention_mask, cos, sin = inputs
            # Replicate the projection from the module:
            B, S, _ = x.shape
            H = mod.cfg.num_heads
            D = mod.cfg.head_dim
            qkv = mod.qkv(x).view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            from binary_embedding.models.encoder import _apply_rope  # local import to avoid cycles
            q, k = _apply_rope(q, k, cos, sin)
            scores = q @ k.transpose(-2, -1) / math.sqrt(D)
            mask = (attention_mask == 0)[:, None, None, :].expand(B, H, S, S)
            scores = scores.masked_fill(mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            # Per (sample, head): mean Shannon entropy of each row, weighted by
            # whether the query position is real (attention_mask[q]=1).
            real_q = attention_mask[:, None, :, None].expand(B, H, S, S)[..., 0]
            ent = -torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs)).sum(-1)
            for h in range(H):
                row = ent[:, h, :][real_q[:, h, :].bool()] if real_q.any() else ent[:, h, :].flatten()
                attn_entropies.append(
                    AttentionEntropy(
                        layer=idx, head=h,
                        entropy_mean=float(row.mean()),
                        entropy_p10=float(row.float().quantile(0.10)) if row.numel() > 0 else float("nan"),
                        entropy_p90=float(row.float().quantile(0.90)) if row.numel() > 0 else float("nan"),
                    )
                )
        return _hook

    for i, blk in enumerate(model.layers):
        hooks.append(blk.attn.register_forward_hook(_attn_hook(i)))

    try:
        ids, attn = _build_probe_batch(cache, n_files=n_files, seq_len=seq_len, seed=seed, device=device)
        _ = model(ids, attn, labels=None, return_mlm_logits=False)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    # Activation spectra: PCA over real positions only (drop pads).
    act_specs: list[ActivationSpectrum] = []
    real_mask = (attn == 1).cpu().numpy()
    for i in sorted(layer_outs):
        h = layer_outs[i].float().cpu().numpy()  # (B, S, H)
        flat = h[real_mask]                       # (N_real, H)
        if flat.size == 0:
            continue
        # PCA via SVD on centered data.
        flat = flat - flat.mean(axis=0, keepdims=True)
        s = np.linalg.svd(flat, full_matrices=False, compute_uv=False)
        act_specs.append(
            ActivationSpectrum(
                layer=i,
                effective_rank=_effective_rank(s),
                activation_norm_mean=float(np.linalg.norm(flat, axis=1).mean()),
                activation_norm_std=float(np.linalg.norm(flat, axis=1).std()),
            )
        )

    return act_specs, attn_entropies


# ---------------------------------------------------------------------------
# Bundle to a tidy dict for parquet writing
# ---------------------------------------------------------------------------


def report(
    model: BinaryEncoder,
    cache: TokenCache,
    *,
    n_files: int = 16,
    seq_len: int = 256,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, list[dict[str, Any]]]:
    device = device or next(model.parameters()).device
    weights = [dc.asdict(w) for w in weight_spectra(model)]
    acts, attns = activation_and_attention_diagnostics(
        model, cache, n_files=n_files, seq_len=seq_len, seed=seed, device=device
    )
    return {
        "weight_spectra": [{**w, "shape": list(w["shape"])} for w in weights],
        "activation_spectra": [dc.asdict(a) for a in acts],
        "attention_entropy": [dc.asdict(a) for a in attns],
    }


__all__ = [
    "ActivationSpectrum", "AttentionEntropy", "WeightSpectrum",
    "activation_and_attention_diagnostics", "report", "weight_spectra",
]
