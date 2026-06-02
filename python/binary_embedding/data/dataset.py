"""Cache reader + PyTorch IterableDataset for MLM pretraining.

A cache is `tokens.bin` (memory-mapped) plus `index.parquet` (one row per file).
Same interface for the byte and BPE variants: the difference lives in the
tokens-bin dtype and the special-token IDs (which are the same constants
across all variants by design — see `python/binary_embedding/constants.py`).

This module never loads `tokens.bin` into RAM; it mmaps. Cache files of
hundreds of GB are fine.
"""

from __future__ import annotations

import dataclasses as dc
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from binary_embedding import _native
from binary_embedding.constants import (
    CLS_ID,
    MASK_ID,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    SEP_ID,
    SPECIAL_TOKENS,
)
from binary_embedding.data.cache import Variant, cache_dir, index_parquet_path, tokens_bin_path
from binary_embedding.data.splits import Split

_SPECIAL_IDS_LIST: list[int] = [
    SPECIAL_TOKENS.index(s) for s in SPECIAL_TOKENS
]  # = [0..6]


@dc.dataclass(slots=True)
class TokenCache:
    """Read-side handle over a (variant, split) cache."""

    variant: Variant
    split: Split
    tokens: np.memmap  # shape (total_tokens,)
    sha256: list[str]
    offsets: np.ndarray  # uint64
    n_tokens: np.ndarray  # uint32
    n_bytes: np.ndarray  # uint64
    metadata_table: object  # pyarrow Table; kept for slicing in eval

    @classmethod
    def open(cls, variant: Variant, split: Split, cache_root: Path) -> "TokenCache":
        tpath = tokens_bin_path(cache_root, variant.name, split)
        ipath = index_parquet_path(cache_root, variant.name, split)
        if not tpath.is_file() or not ipath.is_file():
            raise FileNotFoundError(
                f"missing cache for variant={variant.name!r} split={split!r} under {cache_root}; "
                f"build it with binary_embedding.data.cache.build_cache(...)"
            )
        idx = pq.read_table(ipath)
        tokens = np.memmap(tpath, dtype=variant.numpy_dtype, mode="r")
        return cls(
            variant=variant,
            split=split,
            tokens=tokens,
            sha256=idx.column("sha256").to_pylist(),
            offsets=np.asarray(idx.column("offset").to_pylist(), dtype=np.uint64),
            n_tokens=np.asarray(idx.column("n_tokens").to_pylist(), dtype=np.uint32),
            n_bytes=np.asarray(idx.column("n_bytes").to_pylist(), dtype=np.uint64),
            metadata_table=idx,
        )

    def __len__(self) -> int:
        return len(self.sha256)

    def file_window(self, file_idx: int) -> np.ndarray:
        off = int(self.offsets[file_idx])
        n = int(self.n_tokens[file_idx])
        return np.asarray(self.tokens[off : off + n])


# ---------------------------------------------------------------------------
# Sampling + masking
# ---------------------------------------------------------------------------


def sample_window(
    cache: TokenCache,
    file_idx: int,
    seq_len: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Sample one MLM-ready window, including <CLS> + content + <SEP> + padding.

    Layout: [CLS] tok_0 tok_1 ... tok_{m-1} [SEP] [PAD]*  (length seq_len)

    `m = min(seq_len - 2, file_n_tokens)`. If file is shorter than seq_len-2,
    the tail of the window is padded and attention_mask zeros the padded slots.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be ≥ 2 to fit CLS + SEP")
    body_budget = seq_len - 2
    file_tokens = cache.file_window(file_idx).tolist()
    body, body_mask = _native.sample_token_window(
        file_tokens, seq_len=body_budget, pad_id=PAD_ID, seed=seed
    )
    ids = [CLS_ID, *body, SEP_ID]
    mask = [1, *body_mask, 1]
    # If body got padded, the SEP we just added overlaps a pad cell. That's
    # acceptable: SEP is always present, attention_mask says CLS+pad?+SEP.
    # Simpler: drop pad cells between body and SEP.
    if 0 in body_mask:
        # Truncate to first pad position, then re-pad after SEP.
        first_pad = body_mask.index(0)
        ids = [CLS_ID, *body[:first_pad], SEP_ID]
        mask = [1, *([1] * first_pad), 1]
        # Pad to seq_len.
        pad_count = seq_len - len(ids)
        ids += [PAD_ID] * pad_count
        mask += [0] * pad_count
    assert len(ids) == seq_len
    assert len(mask) == seq_len
    return ids, mask


def make_mlm_batch(
    cache: TokenCache,
    file_indices: list[int],
    seq_len: int,
    mask_ratio: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build a single MLM training batch from a list of file indices.

    Returns tensors shaped (batch, seq_len): `input_ids`, `attention_mask`,
    `labels`. Labels = -100 for non-loss positions.
    """
    batch_input: list[list[int]] = []
    batch_attn: list[list[int]] = []
    batch_labels: list[list[int]] = []
    rng = np.random.default_rng(seed)
    for i, fi in enumerate(file_indices):
        per_seed = int(rng.integers(0, 2**63 - 1))
        ids, mask = sample_window(cache, fi, seq_len=seq_len, seed=per_seed)
        masked_ids, labels = _native.mlm_mask(
            input_ids=ids,
            special_ids=_SPECIAL_IDS_LIST,
            mask_id=MASK_ID,
            vocab_size=cache.variant.vocab_size,
            first_real_id=NUM_SPECIAL_TOKENS,
            mask_ratio=mask_ratio,
            seed=per_seed ^ 0xA5A5_5A5A,
        )
        batch_input.append(masked_ids)
        batch_attn.append(mask)
        batch_labels.append(labels)
    return {
        "input_ids": torch.tensor(batch_input, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------


class MLMTokenStreamDataset(IterableDataset):
    """Yields `(input_ids, attention_mask, labels)` triples sampled from the cache.

    Sampling: pick a file uniformly (or weighted by sqrt(n_bytes) — config flag),
    pick a random window inside it, apply MLM masking. Pure I/O; the worker
    decides when to stop (DataLoader iteration count or epoch token budget).
    """

    def __init__(
        self,
        cache: TokenCache,
        seq_len: int,
        mask_ratio: float = 0.30,
        weighted_by_bytes: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if len(cache) == 0:
            raise ValueError("cache is empty")
        self._cache = cache
        self._seq_len = seq_len
        self._mask_ratio = mask_ratio
        self._weighted = weighted_by_bytes
        self._seed = seed

        if weighted_by_bytes:
            w = np.sqrt(np.asarray(cache.n_bytes, dtype=np.float64))
            self._probs = (w / w.sum()).astype(np.float64)
        else:
            self._probs = None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        seed = self._seed + 1_000_003 * worker_id
        rng = np.random.default_rng(seed)
        n = len(self._cache)
        while True:
            if self._probs is None:
                fi = int(rng.integers(0, n))
            else:
                fi = int(rng.choice(n, p=self._probs))
            per_seed = int(rng.integers(0, 2**63 - 1))
            ids, mask = sample_window(
                self._cache, fi, seq_len=self._seq_len, seed=per_seed
            )
            masked_ids, labels = _native.mlm_mask(
                input_ids=ids,
                special_ids=_SPECIAL_IDS_LIST,
                mask_id=MASK_ID,
                vocab_size=self._cache.variant.vocab_size,
                first_real_id=NUM_SPECIAL_TOKENS,
                mask_ratio=self._mask_ratio,
                seed=per_seed ^ 0xC0FFEE,
            )
            yield {
                "input_ids": torch.tensor(masked_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


__all__ = [
    "TokenCache",
    "MLMTokenStreamDataset",
    "sample_window",
    "make_mlm_batch",
    "cache_dir",
]
