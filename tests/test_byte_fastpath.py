"""Tests for the Rust byte fast-path and batched MLM masker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from binary_embedding import _native


def test_byte_fastpath_matches_python_shift(tmp_path: Path) -> None:
    files = []
    rng = np.random.default_rng(7)
    for i in range(5):
        size = int(rng.integers(0, 4096))
        p = tmp_path / f"f{i:02d}.bin"
        p.write_bytes(bytes(rng.integers(0, 256, size=size, dtype=np.uint8).tolist()))
        files.append(p)

    out = tmp_path / "tokens.bin"
    index = _native.read_paths_to_file_as_bytes(
        [str(p) for p in files], str(out), byte_offset=7, chunk_files=2
    )
    arr = np.frombuffer(out.read_bytes(), dtype="<u2")
    assert sum(n for _, n, _ in index) == len(arr)
    for i, (offset, n_tokens, n_bytes) in enumerate(index):
        slc = arr[offset : offset + n_tokens].tolist()
        expected = (np.frombuffer(files[i].read_bytes(), dtype=np.uint8).astype("<u2") + 7).tolist()
        assert slc == expected, f"file {i} mismatch"
        assert n_bytes == files[i].stat().st_size


def test_byte_fastpath_max_bytes(tmp_path: Path) -> None:
    f = tmp_path / "big.bin"
    f.write_bytes(b"\xab" * 10_000)
    out = tmp_path / "out.bin"
    index = _native.read_paths_to_file_as_bytes(
        [str(f)], str(out), byte_offset=7, chunk_files=1, max_bytes=128
    )
    assert len(index) == 1
    _, n_tokens, n_bytes = index[0]
    assert n_tokens == 128
    assert n_bytes == 128
    arr = np.frombuffer(out.read_bytes(), dtype="<u2")
    assert (arr == 0xAB + 7).all()


def test_byte_fastpath_chunk_zero_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError):
        _native.read_paths_to_file_as_bytes(
            [str(f)], str(tmp_path / "out.bin"), byte_offset=7, chunk_files=0
        )


def test_byte_fastpath_empty_paths(tmp_path: Path) -> None:
    out = tmp_path / "empty.bin"
    index = _native.read_paths_to_file_as_bytes([], str(out), byte_offset=7)
    assert index == []
    assert out.stat().st_size == 0


# ---------------------------------------------------------------------------
# mlm_mask_many: batch matches single calls + parallelism is correctness-preserving
# ---------------------------------------------------------------------------


SPECIALS = [0, 1, 2, 3, 4, 5, 6]


def test_mlm_mask_many_matches_single_at_zero_ratio() -> None:
    """At mask_ratio=0 both forms produce: ids unchanged, all labels = -100."""
    rows = [list(range(7, 7 + 64)) for _ in range(8)]
    masked, labels = _native.mlm_mask_many(
        batch_input_ids=rows,
        special_ids=SPECIALS,
        mask_id=6,
        vocab_size=263,
        first_real_id=7,
        mask_ratio=0.0,
        base_seed=0,
    )
    assert masked == rows
    assert all(all(x == -100 for x in row) for row in labels)


def test_mlm_mask_many_reproducible_with_seed() -> None:
    rows = [list(range(7, 7 + 128)) for _ in range(16)]
    a = _native.mlm_mask_many(
        rows, SPECIALS, mask_id=6, vocab_size=263, first_real_id=7,
        mask_ratio=0.30, base_seed=42,
    )
    b = _native.mlm_mask_many(
        rows, SPECIALS, mask_id=6, vocab_size=263, first_real_id=7,
        mask_ratio=0.30, base_seed=42,
    )
    assert a == b


def test_mlm_mask_many_distinct_seeds_distinct_outputs() -> None:
    rows = [list(range(7, 7 + 128)) for _ in range(8)]
    a = _native.mlm_mask_many(
        rows, SPECIALS, mask_id=6, vocab_size=263, first_real_id=7,
        mask_ratio=0.30, base_seed=1,
    )
    b = _native.mlm_mask_many(
        rows, SPECIALS, mask_id=6, vocab_size=263, first_real_id=7,
        mask_ratio=0.30, base_seed=2,
    )
    assert a != b


def test_mlm_mask_many_per_row_seeds_independent() -> None:
    """Two identical rows in the same batch should not produce identical masking."""
    rows = [list(range(7, 7 + 256)), list(range(7, 7 + 256))]
    masked, labels = _native.mlm_mask_many(
        rows, SPECIALS, mask_id=6, vocab_size=263, first_real_id=7,
        mask_ratio=0.30, base_seed=99,
    )
    # Probabilistically the two masked rows should differ.
    assert masked[0] != masked[1] or labels[0] != labels[1]


def test_mlm_mask_many_specials_not_masked() -> None:
    rows = [[0, 4, 5] + list(range(7, 7 + 64))]  # PAD, CLS, SEP, then bytes
    masked, labels = _native.mlm_mask_many(
        rows, SPECIALS, mask_id=6, vocab_size=263, first_real_id=7,
        mask_ratio=1.0, base_seed=0,
    )
    # Specials must be untouched
    assert masked[0][0] == 0
    assert masked[0][1] == 4
    assert masked[0][2] == 5
    assert labels[0][0] == labels[0][1] == labels[0][2] == -100


def test_mlm_mask_many_distribution_approx_30pct() -> None:
    import math

    n_rows = 200
    seq = 1024
    rows = [list(range(7, 7 + seq)) for _ in range(n_rows)]
    _masked, labels = _native.mlm_mask_many(
        rows, SPECIALS, mask_id=6, vocab_size=263, first_real_id=7,
        mask_ratio=0.30, base_seed=1234,
    )
    n_total = n_rows * seq
    n_label = sum(1 for row in labels for x in row if x != -100)
    mean = 0.30 * n_total
    sd = math.sqrt(0.30 * 0.70 * n_total)
    assert abs(n_label - mean) < 5 * sd
