"""Smoke tests for the binary_embedding._native Rust extension.

These tests assume `uv sync` has been run (which builds the Rust extension via
maturin). They use small, in-repo fixtures only — no external data, no HF Hub,
no GPUs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from binary_embedding import _native


@pytest.fixture()
def tmp_binary(tmp_path: Path) -> Path:
    """Write a 4 KiB pseudo-binary fixture; returns the path."""
    p = tmp_path / "fixture.bin"
    payload = bytes(range(256)) * 16  # 4 KiB, all 256 byte values 16x
    p.write_bytes(payload)
    return p


def test_file_size(tmp_binary: Path) -> None:
    assert _native.file_size(str(tmp_binary)) == 4096


def test_read_head(tmp_binary: Path) -> None:
    head = _native.read_head(str(tmp_binary), 16)
    assert head == bytes(range(16))


def test_read_tail(tmp_binary: Path) -> None:
    tail = _native.read_tail(str(tmp_binary), 16)
    assert tail == bytes(range(240, 256))


def test_read_tail_clamps_to_file_size(tmp_binary: Path) -> None:
    tail = _native.read_tail(str(tmp_binary), 1_000_000)
    assert len(tail) == 4096


def test_read_window(tmp_binary: Path) -> None:
    win = _native.read_window(str(tmp_binary), offset=256, n=16)
    assert win == bytes(range(16))  # second copy of 0..15


def test_read_strided_chunks(tmp_binary: Path) -> None:
    chunks = _native.read_strided_chunks(str(tmp_binary), n_chunks=3, chunk_size=16)
    assert len(chunks) == 3
    assert all(len(c) == 16 for c in chunks)
    # First chunk starts at offset 0.
    assert chunks[0] == bytes(range(16))


def test_read_random_window_deterministic(tmp_binary: Path) -> None:
    a = _native.read_random_window(str(tmp_binary), window=64, seed=42)
    b = _native.read_random_window(str(tmp_binary), window=64, seed=42)
    assert a == b


def test_read_random_windows_count(tmp_binary: Path) -> None:
    windows = _native.read_random_windows(
        str(tmp_binary), n_windows=5, window=64, seed=7
    )
    assert len(windows) == 5
    assert all(len(payload) == 64 for _offset, payload in windows)


def test_sample_token_window_pads_short_streams() -> None:
    ids, mask = _native.sample_token_window([1, 2, 3], seq_len=8, pad_id=0, seed=0)
    assert ids == [1, 2, 3, 0, 0, 0, 0, 0]
    assert mask == [1, 1, 1, 0, 0, 0, 0, 0]


def test_mlm_mask_reproducible() -> None:
    ids = list(range(7, 7 + 64))  # all "real" bytes (no specials)
    out1 = _native.mlm_mask(
        input_ids=ids,
        special_ids=[0, 1, 2, 3, 4, 5, 6],
        mask_id=6,
        vocab_size=263,
        first_real_id=7,
        mask_ratio=0.3,
        seed=99,
    )
    out2 = _native.mlm_mask(
        input_ids=ids,
        special_ids=[0, 1, 2, 3, 4, 5, 6],
        mask_id=6,
        vocab_size=263,
        first_real_id=7,
        mask_ratio=0.3,
        seed=99,
    )
    assert out1 == out2
    masked_ids, labels = out1
    assert len(masked_ids) == len(ids) == len(labels)
    # At least *some* positions get a label.
    assert any(label != -100 for label in labels)
