"""Small ModernBERT-style encoder for the binary-embedding-paper experiments.

Architectural choices (see docs/02-model-architecture.md for the rationale):

- 8 layers, hidden 384, 6 heads (head_dim 64), GeGLU FFN, RMSNorm pre-norm.
- RoPE positional encoding (no learned position embeddings).
- Tied input + MLM head weights.
- bf16-friendly: no LayerNorm fp32 casts, no `eps=1e-12`, etc.

We do *not* use FlashAttention 2 directly here — torch's
`scaled_dot_product_attention` autodetects FA2 / mem-efficient when available
and falls back to the math kernel otherwise. That keeps this file CPU-debuggable.
"""

from __future__ import annotations

import dataclasses as dc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dc.dataclass(frozen=True, slots=True)
class EncoderConfig:
    vocab_size: int
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 6
    ffn_multiplier_num: int = 8
    ffn_multiplier_den: int = 3  # 8/3 GeGLU expansion (LLaMA convention)
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 2
    cls_token_id: int = 4
    sep_token_id: int = 5
    mask_token_id: int = 6
    cls_pool_dim: int = 256
    initializer_range: float = 0.02
    init_scheme: str = "trunc_normal"  # trunc_normal | xavier | scaled_residual
    dropout: float = 0.0
    embedding_dropout: float = 0.0
    hidden_dropout: float = 0.0  # post-attention / post-ffn residual dropout
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0  # stochastic depth (linear schedule across layers)
    layer_scale_init: float | None = None  # None = no LayerScale
    grad_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return self.hidden_size * self.ffn_multiplier_num // self.ffn_multiplier_den


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm without bias. Pre-norm everywhere."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bf16-safe: compute the norm in fp32, then cast back.
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normed * self.weight


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-computed cos/sin for rotary embeddings."""
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("p,d->pd", positions, inv_freq)
    # shape: (seq_len, head_dim)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    return cos, sin


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to q,k of shape (B, H, S, D)."""

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    cos = cos[None, None, :, :]  # (1,1,S,D)
    sin = sin[None, None, :, :]
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.qkv = nn.Linear(H, 3 * H, bias=False)
        self.out = nn.Linear(H, H, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H = self.cfg.num_heads
        D = self.cfg.head_dim

        qkv = self.qkv(x).view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, cos, sin)

        # attention_mask: (B, S) with 1 = real, 0 = pad. SDPA wants bool key-padding.
        key_padding_mask = attention_mask == 0  # True where padding
        bool_mask = key_padding_mask[:, None, None, :].expand(B, H, S, S)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=~bool_mask,
            dropout_p=self.cfg.attention_dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, S, H * D)
        return self.out(attn)


class _LayerScale(nn.Module):
    """Per-feature scaling parameter (CaiT). No-op when init=None."""

    def __init__(self, hidden_size: int, init: float | None) -> None:
        super().__init__()
        if init is None:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.full((hidden_size,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma is None:
            return x
        return x * self.gamma


class _DropPath(nn.Module):
    """Stochastic depth — drop the entire residual branch with probability p."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x.div(keep) * mask


class GeGLU(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ffn = GeGLU(cfg)
        self.attn_scale = _LayerScale(cfg.hidden_size, cfg.layer_scale_init)
        self.ffn_scale = _LayerScale(cfg.hidden_size, cfg.layer_scale_init)
        self.drop_attn = _DropPath(drop_path_rate)
        self.drop_ffn = _DropPath(drop_path_rate)
        self.hidden_dropout = nn.Dropout(cfg.hidden_dropout) if cfg.hidden_dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), attention_mask, cos, sin)
        x = x + self.drop_attn(self.attn_scale(self.hidden_dropout(attn_out)))
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop_ffn(self.ffn_scale(self.hidden_dropout(ffn_out)))
        return x


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------


@dc.dataclass(slots=True)
class EncoderOutputs:
    hidden_states: torch.Tensor  # (B, S, H)
    cls_embedding: torch.Tensor  # (B, cls_pool_dim) — L2-normalized
    mlm_logits: torch.Tensor | None = None  # (B, S, vocab_size) when MLM head ran
    loss: torch.Tensor | None = None  # scalar; only when labels provided


class BinaryEncoder(nn.Module):
    """Encoder-only transformer with weight-tied MLM head and CLS pooler."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.embedding_dropout = (
            nn.Dropout(cfg.embedding_dropout) if cfg.embedding_dropout > 0 else nn.Identity()
        )
        # Linear DropPath schedule across depth (deeper layers drop more often).
        if cfg.drop_path_rate > 0:
            dp_rates = [cfg.drop_path_rate * i / max(1, cfg.num_layers - 1) for i in range(cfg.num_layers)]
        else:
            dp_rates = [0.0] * cfg.num_layers
        self.layers = nn.ModuleList([EncoderBlock(cfg, drop_path_rate=dp) for dp in dp_rates])
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # Weight-tied MLM head: we reuse the embedding matrix at the end.
        self.cls_pool = nn.Linear(cfg.hidden_size, cfg.cls_pool_dim, bias=False)

        self._rope_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.initializer_range
        scheme = self.cfg.init_scheme
        if scheme not in {"trunc_normal", "xavier", "scaled_residual"}:
            raise ValueError(f"unknown init_scheme {scheme!r}")
        # First pass: identical to before.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()
        # Second pass: scaled-residual init (a.k.a. NanoGPT/LLaMA scale-down)
        # divides the residual-out projections by sqrt(2 * num_layers) so the
        # sum-of-residuals doesn't explode at depth.
        if scheme == "scaled_residual":
            scale = 1.0 / math.sqrt(2.0 * self.cfg.num_layers)
            for blk in self.layers:
                with torch.no_grad():
                    blk.attn.out.weight.mul_(scale)
                    blk.ffn.w_down.weight.mul_(scale)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        if not exclude_embeddings:
            return sum(p.numel() for p in self.parameters())
        return sum(
            p.numel()
            for n, p in self.named_parameters()
            if not n.startswith("embed.")
        )

    def _rope(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = _build_rope_cache(
                seq_len,
                self.cfg.head_dim,
                base=self.cfg.rope_theta,
                device=device,
                dtype=dtype,
            )
        cos, sin = self._rope_cache[key]
        return cos.to(dtype), sin.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, S) long
        attention_mask: torch.Tensor,  # (B, S) long {0,1}
        labels: torch.Tensor | None = None,  # (B, S) long, -100 = ignore
        return_mlm_logits: bool = True,
    ) -> EncoderOutputs:
        B, S = input_ids.shape
        x = self.embedding_dropout(self.embed(input_ids))
        cos, sin = self._rope(S, x.dtype, x.device)
        for layer in self.layers:
            if self.cfg.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, cos, sin, use_reentrant=False
                )
            else:
                x = layer(x, attention_mask, cos, sin)
        x = self.final_norm(x)

        # CLS pool: take position 0 (we always prepend <|cls|>).
        cls_h = x[:, 0]
        cls = self.cls_pool(cls_h)
        cls = F.normalize(cls, p=2, dim=-1)

        out_logits = None
        loss = None
        if return_mlm_logits or labels is not None:
            # Tied head: logits = x @ embed.weight.T
            logits = x @ self.embed.weight.t()
            out_logits = logits if return_mlm_logits else None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, self.cfg.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )
        return EncoderOutputs(
            hidden_states=x, cls_embedding=cls, mlm_logits=out_logits, loss=loss
        )

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: forward without MLM head, return L2-normalized CLS only."""
        self.eval()
        out = self.forward(input_ids, attention_mask, labels=None, return_mlm_logits=False)
        return out.cls_embedding


def small_encoder_config(vocab_size: int, *, max_seq_len: int = 2048) -> EncoderConfig:
    """Headline ~14 M-backbone config (8L × 384h × 6 heads × 8/3 GeGLU).

    Per docs/02-model-architecture.md. Used in Phase A.
    """
    return EncoderConfig(vocab_size=vocab_size, max_seq_len=max_seq_len)


def tiny_encoder_config(vocab_size: int, *, max_seq_len: int = 2048) -> EncoderConfig:
    """~3.5 M-backbone config (4L × 256h × 4 heads × 8/3 GeGLU).

    Smallest cell of the Phase B architecture cube — fast iteration / seed-noise
    bound. Same head_dim=64 as the larger configs to keep attention shape sensible.
    """
    return EncoderConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )


def medium_encoder_config(vocab_size: int, *, max_seq_len: int = 2048) -> EncoderConfig:
    """~38.5 M-backbone config (12L × 512h × 8 heads × 8/3 GeGLU).

    Largest cell of the Phase B architecture cube — same head_dim=64 as small/tiny.
    Tests whether the byte-vs-BPE / vocab-size effects scale or invert with depth+width.
    """
    return EncoderConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
    )


@dc.dataclass(slots=True)
class WeightTyingAudit:
    """Result of `verify_weight_tying`. `is_tied` is the bottom-line yes/no."""

    embed_param_name: str
    embed_storage_id: int
    n_vocab_sized_tensors: int
    has_separate_mlm_head: bool
    mlm_logits_depend_on_embed: bool

    @property
    def is_tied(self) -> bool:
        return (
            self.n_vocab_sized_tensors == 1
            and not self.has_separate_mlm_head
            and self.mlm_logits_depend_on_embed
        )


def verify_weight_tying(model: "BinaryEncoder") -> WeightTyingAudit:
    """Assert the encoder uses tied embeddings end-to-end (no separate MLM head).

    Three checks:

    1. Exactly one trainable tensor has the embedding shape `(vocab_size, hidden_size)`.
    2. No submodule named `mlm_head` / `lm_head` / `output_projection` etc. exists.
    3. A gradient on the embedding flows to the MLM logits — i.e. the forward
       actually uses `self.embed.weight` to compute logits.
    """
    cfg = model.cfg
    embed_shape = (cfg.vocab_size, cfg.hidden_size)
    matches = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and tuple(p.shape) == embed_shape
    ]
    suspect_names = {"mlm_head", "lm_head", "output_projection", "decoder", "head"}
    has_separate = any(
        any(part in n for part in suspect_names) for n, _ in model.named_modules()
    )

    embed_name, embed_param = next(
        ((n, p) for n, p in model.named_parameters() if n == "embed.weight"),
        (matches[0][0], matches[0][1]) if matches else ("?", None),
    )

    # Forward with require_grad on the embedding, ensure MLM logits depend on it.
    grad_flows = False
    if embed_param is not None:
        was_training = model.training
        model.eval()
        try:
            B, S = 2, 8
            ids = torch.zeros(B, S, dtype=torch.long, device=embed_param.device)
            ids[:, 0] = cfg.cls_token_id
            attn = torch.ones_like(ids)
            # Compute MLM logits and a synthetic loss; if embed grad becomes non-zero,
            # MLM logits depend on embed.weight.
            embed_param.requires_grad_(True)
            if embed_param.grad is not None:
                embed_param.grad = None
            out = model(ids, attn, labels=None, return_mlm_logits=True)
            assert out.mlm_logits is not None
            out.mlm_logits.sum().backward()
            grad_flows = bool(
                embed_param.grad is not None and embed_param.grad.abs().sum().item() > 0
            )
            embed_param.grad = None
        finally:
            if was_training:
                model.train()

    return WeightTyingAudit(
        embed_param_name=embed_name,
        embed_storage_id=id(embed_param) if embed_param is not None else -1,
        n_vocab_sized_tensors=len(matches),
        has_separate_mlm_head=has_separate,
        mlm_logits_depend_on_embed=grad_flows,
    )


__all__ = [
    "EncoderConfig", "BinaryEncoder", "EncoderOutputs",
    "WeightTyingAudit",
    "tiny_encoder_config", "small_encoder_config", "medium_encoder_config",
    "verify_weight_tying",
]
