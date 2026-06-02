"""Tests for `binary_embedding.models.encoder.BinaryEncoder`."""

from __future__ import annotations

import pytest
import torch

from binary_embedding.constants import (
    BYTE_VOCAB_SIZE,
    CLS_ID,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    SEP_ID,
)
from binary_embedding.models.encoder import (
    BinaryEncoder,
    EncoderConfig,
    small_encoder_config,
)


def _toy_cfg(vocab: int = BYTE_VOCAB_SIZE) -> EncoderConfig:
    # Deliberately tiny so tests run in milliseconds.
    return EncoderConfig(
        vocab_size=vocab, hidden_size=32, num_layers=2, num_heads=4,
        ffn_multiplier_num=4, ffn_multiplier_den=1, max_seq_len=128,
        cls_pool_dim=16,
    )


def _make_batch(cfg: EncoderConfig, batch: int, seq: int) -> dict[str, torch.Tensor]:
    ids = torch.randint(low=NUM_SPECIAL_TOKENS, high=cfg.vocab_size, size=(batch, seq))
    ids[:, 0] = cfg.cls_token_id
    ids[:, -1] = cfg.sep_token_id
    attn = torch.ones(batch, seq, dtype=torch.long)
    labels = torch.full_like(ids, fill_value=-100)
    labels[:, 1::2] = ids[:, 1::2]
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def test_forward_shapes() -> None:
    cfg = _toy_cfg()
    model = BinaryEncoder(cfg)
    batch = _make_batch(cfg, 4, 32)
    out = model(**batch)
    assert out.hidden_states.shape == (4, 32, cfg.hidden_size)
    assert out.cls_embedding.shape == (4, cfg.cls_pool_dim)
    assert out.mlm_logits is not None
    assert out.mlm_logits.shape == (4, 32, cfg.vocab_size)
    assert out.loss is not None
    assert out.loss.ndim == 0


def test_cls_embedding_is_unit_norm() -> None:
    cfg = _toy_cfg()
    model = BinaryEncoder(cfg)
    batch = _make_batch(cfg, 4, 16)
    out = model(**batch)
    norms = out.cls_embedding.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_loss_is_log_vocab_at_init() -> None:
    """At random init the MLM loss should be roughly ln(vocab_size)."""
    import math

    cfg = _toy_cfg(vocab=263)
    torch.manual_seed(0)
    model = BinaryEncoder(cfg)
    batch = _make_batch(cfg, 16, 64)
    out = model(**batch)
    expected = math.log(cfg.vocab_size)
    assert abs(out.loss.item() - expected) < 1.0


def test_backward_populates_grads() -> None:
    cfg = _toy_cfg()
    model = BinaryEncoder(cfg)
    batch = _make_batch(cfg, 2, 16)
    out = model(**batch)
    out.loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 5


def test_padding_mask_blocks_attention() -> None:
    """Tokens at attention_mask==0 must not influence non-pad outputs."""
    cfg = _toy_cfg()
    torch.manual_seed(0)
    model = BinaryEncoder(cfg).eval()
    base = _make_batch(cfg, 1, 32)
    base["attention_mask"][0, 16:] = 0  # right-pad the second half
    base["input_ids"][0, 16:] = PAD_ID

    with torch.no_grad():
        out_a = model(base["input_ids"], base["attention_mask"], return_mlm_logits=False)

    # Now scribble random ids into the padded positions; output for non-pad must be unchanged.
    scribbled = base["input_ids"].clone()
    scribbled[0, 16:] = torch.randint(NUM_SPECIAL_TOKENS, cfg.vocab_size, (16,))
    with torch.no_grad():
        out_b = model(scribbled, base["attention_mask"], return_mlm_logits=False)

    diff = (out_a.hidden_states[0, :16] - out_b.hidden_states[0, :16]).abs().max().item()
    assert diff < 1e-4, f"padding leaked into non-pad outputs (max diff {diff})"


def test_encode_returns_l2_normalized_cls() -> None:
    cfg = _toy_cfg()
    model = BinaryEncoder(cfg)
    batch = _make_batch(cfg, 3, 16)
    cls = model.encode(batch["input_ids"], batch["attention_mask"])
    assert cls.shape == (3, cfg.cls_pool_dim)
    assert torch.allclose(cls.norm(dim=-1), torch.ones(3), atol=1e-3)


def test_pad_embedding_zeroed_at_init() -> None:
    cfg = _toy_cfg()
    model = BinaryEncoder(cfg)
    pad_emb = model.embed.weight[cfg.pad_token_id]
    assert torch.allclose(pad_emb, torch.zeros_like(pad_emb))


def test_param_counts_for_paper_configs() -> None:
    """Sanity: backbone is identical across variants; embedding scales with vocab."""
    backbones = []
    for vsize in (263, 4_103, 16_391, 65_543):
        cfg = small_encoder_config(vocab_size=vsize)
        m = BinaryEncoder(cfg)
        backbones.append(m.num_parameters(exclude_embeddings=True))
    # All identical
    assert len(set(backbones)) == 1
    # Headline backbone is ~14M (8L × 384h × GeGLU 8/3) — see docs/02.
    assert 12_000_000 < backbones[0] < 16_000_000


def test_bf16_forward_runs() -> None:
    cfg = _toy_cfg()
    model = BinaryEncoder(cfg).to(torch.bfloat16)
    batch = _make_batch(cfg, 2, 16)
    out = model(batch["input_ids"], batch["attention_mask"], return_mlm_logits=False)
    assert out.cls_embedding.dtype == torch.bfloat16


@pytest.mark.parametrize("seq_len", [16, 32, 64, 128])
def test_variable_seq_len_works(seq_len: int) -> None:
    cfg = _toy_cfg()
    model = BinaryEncoder(cfg)
    batch = _make_batch(cfg, 2, seq_len)
    out = model(**batch)
    assert out.hidden_states.shape == (2, seq_len, cfg.hidden_size)
