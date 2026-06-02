"""Statistical validation of `_native.mlm_mask`.

We pre-register expectations: at mask_ratio=0.30 with the canonical 80/10/10
keep_random_split, over a long stream of non-special tokens we expect

- ~30 % of positions to receive a label (i.e., be selected for the loss)
- among labelled positions: 80 % become MASK, 10 % become a random in-vocab
  token, 10 % stay unchanged.

We use generous tolerances (3 sigma) appropriate for a single statistical run.
"""

from __future__ import annotations

import math
import random

import pytest

from binary_embedding import _native


SPECIAL_IDS = list(range(7))  # IDs 0..6 = our specials by convention
MASK_ID = 6  # <|mask|>
PAD_ID = 2
FIRST_REAL_ID = 7  # bytes start at 7 in the byte vocab; specials below


def test_mlm_no_specials_get_masked() -> None:
    # Mostly bytes, but stuff a few CLS/SEP/PAD tokens in.
    rng = random.Random(0)
    stream = [rng.randrange(FIRST_REAL_ID, 263) for _ in range(2000)]
    for i in (0, 7, 13, 100, 1500):
        stream[i] = i % len(SPECIAL_IDS)
    masked, labels = _native.mlm_mask(
        input_ids=stream,
        special_ids=SPECIAL_IDS,
        mask_id=MASK_ID,
        vocab_size=263,
        first_real_id=FIRST_REAL_ID,
        mask_ratio=0.30,
        seed=42,
    )
    for i in (0, 7, 13, 100, 1500):
        # special positions: token must remain unchanged and have label -100
        assert masked[i] == stream[i]
        assert labels[i] == -100


def test_mlm_distribution_at_30pct() -> None:
    n = 200_000
    rng = random.Random(123)
    stream = [rng.randrange(FIRST_REAL_ID, 263) for _ in range(n)]
    masked, labels = _native.mlm_mask(
        input_ids=stream,
        special_ids=SPECIAL_IDS,
        mask_id=MASK_ID,
        vocab_size=263,
        first_real_id=FIRST_REAL_ID,
        mask_ratio=0.30,
        seed=999,
    )
    n_label = sum(1 for x in labels if x != -100)
    # 30% mask ratio over n positions: binomial mean=0.3n, std=sqrt(0.21n)
    mean, std = 0.30 * n, math.sqrt(0.30 * 0.70 * n)
    assert abs(n_label - mean) < 5 * std, f"label fraction off: got {n_label}, expected ~{mean:.0f}"

    # Among labelled positions, count {mask, random, keep}
    n_mask = n_random = n_keep = 0
    for i, lab in enumerate(labels):
        if lab == -100:
            continue
        orig = lab
        cur = masked[i]
        if cur == MASK_ID:
            n_mask += 1
        elif cur == orig:
            n_keep += 1
        else:
            n_random += 1
    total = n_mask + n_random + n_keep
    assert total == n_label
    # Tolerances at 5 sigma against expected sub-binomials.
    for got, p, name in [(n_mask, 0.80, "mask"), (n_random, 0.10, "random"), (n_keep, 0.10, "keep")]:
        mu = p * total
        sd = math.sqrt(p * (1 - p) * total)
        assert abs(got - mu) < 5 * sd, f"{name} fraction off: got {got}, expected ~{mu:.0f} ± {sd:.0f}"


def test_mlm_seed_changes_outcome() -> None:
    rng = random.Random(5)
    stream = [rng.randrange(FIRST_REAL_ID, 263) for _ in range(5000)]
    a = _native.mlm_mask(
        input_ids=stream,
        special_ids=SPECIAL_IDS,
        mask_id=MASK_ID,
        vocab_size=263,
        first_real_id=FIRST_REAL_ID,
        mask_ratio=0.30,
        seed=1,
    )
    b = _native.mlm_mask(
        input_ids=stream,
        special_ids=SPECIAL_IDS,
        mask_id=MASK_ID,
        vocab_size=263,
        first_real_id=FIRST_REAL_ID,
        mask_ratio=0.30,
        seed=2,
    )
    assert a != b


def test_mlm_seed_repeats_outcome() -> None:
    rng = random.Random(5)
    stream = [rng.randrange(FIRST_REAL_ID, 263) for _ in range(5000)]
    a = _native.mlm_mask(
        input_ids=stream,
        special_ids=SPECIAL_IDS,
        mask_id=MASK_ID,
        vocab_size=263,
        first_real_id=FIRST_REAL_ID,
        mask_ratio=0.30,
        seed=42,
    )
    b = _native.mlm_mask(
        input_ids=stream,
        special_ids=SPECIAL_IDS,
        mask_id=MASK_ID,
        vocab_size=263,
        first_real_id=FIRST_REAL_ID,
        mask_ratio=0.30,
        seed=42,
    )
    assert a == b


def test_mlm_zero_ratio_is_noop() -> None:
    stream = list(range(FIRST_REAL_ID, FIRST_REAL_ID + 64))
    masked, labels = _native.mlm_mask(
        input_ids=stream,
        special_ids=SPECIAL_IDS,
        mask_id=MASK_ID,
        vocab_size=263,
        first_real_id=FIRST_REAL_ID,
        mask_ratio=0.0,
        seed=0,
    )
    assert masked == stream
    assert labels == [-100] * len(stream)


def test_mlm_invalid_split_raises() -> None:
    with pytest.raises(ValueError):
        _native.mlm_mask(
            input_ids=[7, 8, 9],
            special_ids=SPECIAL_IDS,
            mask_id=MASK_ID,
            vocab_size=263,
            first_real_id=FIRST_REAL_ID,
            mask_ratio=0.30,
            seed=0,
            keep_random_split=(0.5, 0.4, 0.4),
        )


def test_mlm_invalid_ratio_raises() -> None:
    with pytest.raises(ValueError):
        _native.mlm_mask(
            input_ids=[7, 8, 9],
            special_ids=SPECIAL_IDS,
            mask_id=MASK_ID,
            vocab_size=263,
            first_real_id=FIRST_REAL_ID,
            mask_ratio=1.5,
            seed=0,
        )
