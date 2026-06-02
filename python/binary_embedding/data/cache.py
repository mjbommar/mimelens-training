"""Pre-encode the corpus to a packed `tokens.bin` + sidecar `index.parquet`.

One cache per (variant, split). Variants:

- `byte`        — raw bytes shifted by NUM_SPECIAL_TOKENS=7 (no tokenizer).
- `bpe-{N}k`    — one of the binary-tokenizer-001-* tokenizers, N in {4, 8, 16, 32, 64}.

Output layout (per cache):

    cache_root/<variant>/<split>/
    ├── tokens.bin         # packed u16 (vocab ≤ 32K) or u32 (vocab > 32K), little-endian
    └── index.parquet      # one row per file: sha256, offset, n_tokens, n_bytes, plus metadata

Read side: memory-map `tokens.bin` and slice using offset/n_tokens from the
parquet. See `dataset.py`.
"""

from __future__ import annotations

import dataclasses as dc
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from binary_embedding import _native
from binary_embedding.constants import (
    BYTE_OFFSET,
    BYTE_VOCAB_SIZE,
    HF_TOKENIZERS,
    NUM_SPECIAL_TOKENS,
)
from binary_embedding.data.splits import Record, Split, load_records

# Local copies of the binary-BPE tokenizer JSONs (no network at corpus build
# time). Point MIMELENS_TOKENIZER_DIR at a directory holding the JSONs in the
# layout below, or download them from the Hugging Face Hub
# (mjbommar/binary-tokenizer-001-*; see HF_TOKENIZERS in constants.py).
_TOKENIZER_DIR = Path(os.environ.get("MIMELENS_TOKENIZER_DIR", "tokenizers"))
LOCAL_TOKENIZER_PATHS: dict[int, Path] = {
    4_096: _TOKENIZER_DIR / "tokenizer-4k" / "tokenizer-4096.json",
    8_192: _TOKENIZER_DIR / "tokenizer-8k" / "tokenizer-8192.json",
    16_384: _TOKENIZER_DIR / "tokenizer-16k" / "tokenizer-16384.json",
    32_768: _TOKENIZER_DIR / "tokenizer-32k" / "tokenizer-32768.json",
    65_536: _TOKENIZER_DIR / "tokenizer-64k" / "tokenizer-65536.json",
}

VariantName = str  # e.g. "byte", "bpe-4k", "bpe-16k"


@dc.dataclass(frozen=True, slots=True)
class Variant:
    """The input pipeline that defines what each token IS."""

    name: VariantName
    vocab_size: int
    is_byte: bool
    tokenizer_path: Path | None  # None for the byte variant
    dtype: Literal["u16", "u32"]

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype("<u2") if self.dtype == "u16" else np.dtype("<u4")

    @property
    def first_real_id(self) -> int:
        """First non-special token id."""
        return NUM_SPECIAL_TOKENS  # specials are always 0..6


def variant_byte() -> Variant:
    return Variant(
        name="byte",
        vocab_size=BYTE_VOCAB_SIZE,
        is_byte=True,
        tokenizer_path=None,
        dtype="u16",  # 263 ids fits in u16; we keep all caches u16-aligned where possible
    )


def variant_bpe(vocab_size: int) -> Variant:
    if vocab_size not in HF_TOKENIZERS:
        raise ValueError(f"unknown bpe vocab_size {vocab_size}; expected one of {sorted(HF_TOKENIZERS)}")
    path = LOCAL_TOKENIZER_PATHS.get(vocab_size)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"local tokenizer.json missing for vocab={vocab_size} at {path}; "
            f"either populate it or download {HF_TOKENIZERS[vocab_size]} from HF"
        )
    # vocab=64k actually lives in 65,536 + 7 specials -> can exceed u16 in encoded ids.
    # Use u32 there. All others fit u16.
    use_u32 = vocab_size > 32_768
    return Variant(
        name=f"bpe-{vocab_size // 1024}k" if vocab_size >= 1024 else f"bpe-{vocab_size}",
        vocab_size=vocab_size + NUM_SPECIAL_TOKENS,
        is_byte=False,
        tokenizer_path=path,
        dtype="u32" if use_u32 else "u16",
    )


def known_variants() -> list[Variant]:
    return [variant_byte()] + [variant_bpe(v) for v in sorted(HF_TOKENIZERS)]


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def cache_dir(root: Path, variant: VariantName, split: Split) -> Path:
    return Path(root) / variant / split


def tokens_bin_path(root: Path, variant: VariantName, split: Split) -> Path:
    return cache_dir(root, variant, split) / "tokens.bin"


def index_parquet_path(root: Path, variant: VariantName, split: Split) -> Path:
    return cache_dir(root, variant, split) / "index.parquet"


@dc.dataclass(frozen=True, slots=True)
class CacheManifest:
    """Per-cache summary written to MANIFEST.json (human-readable, also a guard)."""

    variant: VariantName
    split: Split
    vocab_size: int
    dtype: str
    n_files: int
    n_tokens: int
    n_bytes: int
    on_disk_bytes: int


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _filter_existing(records: Iterable[Record]) -> tuple[list[Record], int]:
    keep: list[Record] = []
    missing = 0
    for r in records:
        if r.exists():
            keep.append(r)
        else:
            missing += 1
    return keep, missing


def _index_table(records: list, offsets: list[int], n_tokens: list[int]) -> pa.Table:
    """Build the per-row index parquet.

    Accepts both `data.splits.Record` (single-source path) and
    `data.sources.TrainingRecord` (multi-source path). Fields the record
    doesn't carry get a sensible default. The `source` column is always
    populated; for legacy `Record` instances it falls back to "binary-30k".
    """

    def _g(r, name: str, default):
        return getattr(r, name, default)

    return pa.table(
        {
            "sha256": pa.array([r.sha256 for r in records], type=pa.string()),
            "offset": pa.array(offsets, type=pa.uint64()),
            "n_tokens": pa.array(n_tokens, type=pa.uint32()),
            "n_bytes": pa.array([r.file_size for r in records], type=pa.uint64()),
            "source": pa.array([_g(r, "source", "binary-30k") for r in records], type=pa.string()),
            "platform": pa.array([_g(r, "platform", "") for r in records], type=pa.string()),
            "file_format": pa.array([_g(r, "file_format", "") for r in records], type=pa.string()),
            "architecture": pa.array([_g(r, "architecture", "") for r in records], type=pa.string()),
            "binary_type": pa.array([_g(r, "binary_type", "") for r in records], type=pa.string()),
            "is_malware": pa.array([_g(r, "is_malware", False) for r in records], type=pa.bool_()),
            "is_packed": pa.array([_g(r, "is_packed", False) for r in records], type=pa.bool_()),
            "is_signed": pa.array([_g(r, "is_signed", False) for r in records], type=pa.bool_()),
            "is_stripped": pa.array([_g(r, "is_stripped", False) for r in records], type=pa.bool_()),
            "entropy": pa.array([_g(r, "entropy", 0.0) for r in records], type=pa.float32()),
            "mime_type": pa.array([_g(r, "mime_type", "") for r in records], type=pa.string()),
        }
    )


def _build_byte_cache(
    variant: Variant,
    records: list[Record],
    out_dir: Path,
    *,
    chunk_files: int = 64,
    max_bytes_per_file: int | None = None,
) -> CacheManifest:
    """Streaming-write the byte variant via _native.read_paths_to_file_as_bytes.

    Rust does the file reads + shift + write in rayon-parallel chunks. Returns
    the same per-file index shape as the BPE path, so the index parquet builder
    is shared.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = out_dir / "tokens.bin"
    paths = [r.file_path for r in records]
    index = _native.read_paths_to_file_as_bytes(
        paths,
        str(tokens_path),
        byte_offset=BYTE_OFFSET,
        chunk_files=chunk_files,
        max_bytes=max_bytes_per_file,
    )
    offsets = [int(o) for (o, _, _) in index]
    n_tokens = [int(n) for (_, n, _) in index]
    total_bytes = sum(int(b) for (_, _, b) in index)
    pq.write_table(_index_table(records, offsets, n_tokens), out_dir / "index.parquet")
    return CacheManifest(
        variant=variant.name,
        split=getattr(records[0], "split", "train") if records else "train",
        vocab_size=variant.vocab_size,
        dtype=variant.dtype,
        n_files=len(records),
        n_tokens=sum(n_tokens),
        n_bytes=total_bytes,
        on_disk_bytes=tokens_path.stat().st_size,
    )


def _build_bpe_cache(
    variant: Variant,
    records: list[Record],
    out_dir: Path,
    *,
    chunk_files: int = 64,
    max_bytes_per_file: int | None = None,
) -> CacheManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = out_dir / "tokens.bin"
    if variant.tokenizer_path is None:  # pragma: no cover
        raise AssertionError("variant_bpe must have tokenizer_path")
    tok = _native.BinaryTokenizer.from_file(str(variant.tokenizer_path))
    paths = [r.file_path for r in records]
    index = _native.tokenize_paths_to_file(
        tok,
        paths,
        str(tokens_path),
        dtype=variant.dtype,
        chunk_files=chunk_files,
        max_bytes=max_bytes_per_file,
    )
    offsets = [int(o) for (o, _, _) in index]
    n_tokens = [int(n) for (_, n, _) in index]
    pq.write_table(_index_table(records, offsets, n_tokens), out_dir / "index.parquet")
    total_tokens = sum(n_tokens)
    total_bytes = sum(r.file_size for r in records)
    return CacheManifest(
        variant=variant.name,
        split=getattr(records[0], "split", "train") if records else "train",
        vocab_size=variant.vocab_size,
        dtype=variant.dtype,
        n_files=len(records),
        n_tokens=total_tokens,
        n_bytes=total_bytes,
        on_disk_bytes=tokens_path.stat().st_size,
    )


def build_cache(
    variant: Variant,
    split: Split,
    cache_root: Path,
    *,
    chunk_files: int = 64,
    max_bytes_per_file: int | None = None,
    require_exists: bool = True,
    limit: int | None = None,
) -> CacheManifest:
    """Backwards-compat wrapper: build one (variant, split) cache from binary-30k.

    For multi-source builds use `build_mixed_cache(...)` directly.
    """
    from binary_embedding.data.sources import Binary30kSource

    return build_mixed_cache(
        variant=variant, split=split, cache_root=cache_root,
        sources=[Binary30kSource(require_exists=require_exists)],
        per_source_limit={"binary-30k": limit} if limit is not None else None,
        chunk_files=chunk_files,
        max_bytes_per_file=max_bytes_per_file,
        require_exists=require_exists,
    )


def build_mixed_cache(
    variant: Variant,
    split: Split,
    cache_root: Path,
    *,
    sources: list,  # list[Source] but kept untyped to avoid an import cycle
    per_source_limit: dict[str, int] | None = None,
    chunk_files: int = 64,
    max_bytes_per_file: int | None = None,
    require_exists: bool = True,
) -> CacheManifest:
    """Build a cache from one or more `data.sources.Source` objects.

    Records are concatenated in source-iteration order, deduped by
    `(source, sha256)`. The resulting `index.parquet` carries a `source`
    column so downstream code can slice / weight per source.

    Cache layout is identical to single-source: `<cache_root>/<variant>/<split>/`.
    Use distinct `cache_root` paths if you want to keep multiple mixes for
    the same variant on disk.
    """
    from binary_embedding.data.sources import mix_records

    records = mix_records(
        sources, split,
        per_source_limit=per_source_limit,
        require_exists=require_exists,
    )
    if not records:
        raise RuntimeError(
            f"no records resolved for split={split} from sources "
            f"{[s.name for s in sources]}"
        )
    out_dir = cache_dir(cache_root, variant.name, split)
    if variant.is_byte:
        return _build_byte_cache(
            variant, records, out_dir,
            chunk_files=chunk_files, max_bytes_per_file=max_bytes_per_file,
        )
    return _build_bpe_cache(
        variant, records, out_dir,
        chunk_files=chunk_files, max_bytes_per_file=max_bytes_per_file,
    )
