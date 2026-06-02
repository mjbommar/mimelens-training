"""Pluggable training-data sources for the cache builder.

Each `Source` exposes `.records(split)` and returns a list of `TrainingRecord`s
(a unified shape across heterogeneous corpora). The cache builder concatenates
records from one or more sources into a single packed `tokens.bin` + sidecar
`index.parquet` that carries a `source` column for downstream slicing.

Two reference sources ship here:

- `Binary30kSource`     — the public ``binary-30k`` corpus (29,793 ELF / PE /
                          Mach-O / APK binaries) with stratified 70/15/15
                          splits, read from local arrow shards via
                          `data.splits`. See the dataset on the Hugging Face
                          Hub: ``mjbommar/binary-30k-tokenized``.
- `LocalDirectorySource`— bring-your-own corpus: walks a directory tree, emits
                          one record per file, and assigns hash-bucketed splits
                          so adding files later never reshuffles existing
                          assignments. This is the extension point for the
                          larger, mixed-provenance corpus used in the paper,
                          which is not redistributable.

To add a source, implement the `Source` protocol (a `.name` and a
`.records(split, *, limit)` method returning `TrainingRecord`s) and pass it to
`cache.build_mixed_cache([...])`. Sources without a canonical split can use
`assign_split_by_hash(records)` for deterministic, sha256-bucketed splits.
"""

from __future__ import annotations

import dataclasses as dc
import hashlib
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from binary_embedding.data.splits import Split

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified training record
# ---------------------------------------------------------------------------


@dc.dataclass(frozen=True, slots=True)
class TrainingRecord:
    """A single file destined for the cache, normalized across sources."""

    sha256: str               # 64-char identifier (real sha256 if available, else hash of path)
    file_path: str
    file_size: int
    source: str               # "binary-30k", "local-dir", ...
    # All metadata below is optional. Index parquet columns get sensible defaults
    # (empty string / 0 / False) for sources that don't supply the field.
    platform: str = ""
    file_format: str = ""
    architecture: str = ""
    binary_type: str = ""
    is_malware: bool = False
    is_packed: bool = False
    is_signed: bool = False
    is_stripped: bool = False
    entropy: float = 0.0
    mime_type: str = ""

    def exists(self) -> bool:
        return Path(self.file_path).is_file()


# ---------------------------------------------------------------------------
# Hash-based split helper for sources without canonical splits
# ---------------------------------------------------------------------------


def _hash_bucket(key: str, modulo: int = 100) -> int:
    """Deterministic 0..modulo-1 bucket for `key`."""
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big") % modulo


def assign_split_by_hash(
    records: Iterable[TrainingRecord],
    *,
    train_pct: int = 90,
    val_pct: int = 5,
) -> dict[str, Split]:
    """Assign each record's sha256 to a split deterministically by hash bucket.

    Returns `{sha256: split}`. Default 90/5/5 — a bit more train-heavy than the
    binary-30k 70/15/15 because bring-your-own corpora tend to be larger (more
    train data is fine; eval is plenty at 5%).
    """
    out: dict[str, Split] = {}
    for r in records:
        b = _hash_bucket(r.sha256)
        if b < train_pct:
            out[r.sha256] = "train"
        elif b < train_pct + val_pct:
            out[r.sha256] = "validation"
        else:
            out[r.sha256] = "test"
    return out


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------


class Source(Protocol):
    """A named training-data source.

    Implementations must:
    - Expose a stable string `.name` (will be the value in the index parquet's
      `source` column).
    - Return all records for a given split.
    - Be safe to call repeatedly (caching internally is fine).
    """

    name: str

    def records(self, split: Split, *, limit: int | None = None) -> list[TrainingRecord]: ...


# ---------------------------------------------------------------------------
# Binary30kSource — the public arrow-shard corpus with canonical splits
# ---------------------------------------------------------------------------


@dc.dataclass(slots=True)
class Binary30kSource:
    """Wraps the binary-30k arrow shards via `data.splits.load_records`.

    Point `data.splits` at your local copy of the shards with the
    ``MIMELENS_BINARY30K_SPLITS`` environment variable (see `data.splits`).
    """

    name: str = "binary-30k"
    require_exists: bool = True

    def records(self, split: Split, *, limit: int | None = None) -> list[TrainingRecord]:
        from binary_embedding.data.splits import load_records

        recs = load_records(split, require_exists=self.require_exists, limit=limit)
        return [
            TrainingRecord(
                sha256=r.sha256,
                file_path=r.file_path,
                file_size=r.file_size,
                source=self.name,
                platform=r.platform,
                file_format=r.file_format,
                architecture=r.architecture,
                binary_type=r.binary_type,
                is_malware=r.is_malware,
                is_packed=r.is_packed,
                is_signed=r.is_signed,
                is_stripped=r.is_stripped,
                entropy=r.entropy,
            )
            for r in recs
        ]


# ---------------------------------------------------------------------------
# LocalDirectorySource — bring-your-own corpus, walks a directory tree
# ---------------------------------------------------------------------------


@dc.dataclass(slots=True)
class LocalDirectorySource:
    """Walk a directory tree and emit one record per file.

    The extension point for any corpus you hold locally and cannot redistribute.
    Splits are hash-bucketed by relative path so adding more files later does
    not reshuffle existing assignments. If a file is named by its sha256 (a
    64-char lowercase hex basename, a common content-addressed layout) that hash
    is used as the record id; otherwise the id is the sha256 of the relative
    path.

    `platform` is set to the first path component under `root`, so a layout like
    ``root/<corpus-name>/...`` lets you slice by sub-corpus downstream.
    """

    root: Path
    name: str = "local-dir"
    train_pct: int = 90
    val_pct: int = 5
    min_size: int = 1024          # skip near-empty placeholder files
    max_size: int | None = None   # cap to avoid memory blowup on giant files
    _cache: list[TrainingRecord] | None = dc.field(default=None, init=False, repr=False)
    _split_map: dict[str, Split] | None = dc.field(default=None, init=False, repr=False)

    def _all_records(self) -> list[TrainingRecord]:
        if self._cache is not None:
            return self._cache
        root = Path(self.root)
        if not root.is_dir():
            log.warning("%s root missing at %s", self.name, root)
            self._cache = []
            self._split_map = {}
            return self._cache
        out: list[TrainingRecord] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size < self.min_size:
                continue
            if self.max_size is not None and size > self.max_size:
                continue
            rel = str(p.relative_to(root))
            sha = hashlib.sha256(rel.encode()).hexdigest()
            base = p.name
            if len(base) == 64 and re.fullmatch(r"[0-9a-f]{64}", base):
                sha = base
            corpus = p.parts[len(root.parts)] if len(p.parts) > len(root.parts) else "?"
            out.append(TrainingRecord(
                sha256=sha,
                file_path=str(p),
                file_size=size,
                source=self.name,
                platform=corpus,
            ))
        self._cache = out
        self._split_map = assign_split_by_hash(
            out, train_pct=self.train_pct, val_pct=self.val_pct
        )
        return out

    def records(self, split: Split, *, limit: int | None = None) -> list[TrainingRecord]:
        all_recs = self._all_records()
        sm = self._split_map or {}
        out = [r for r in all_recs if sm.get(r.sha256) == split]
        if limit is not None:
            out = out[:limit]
        return out


# ---------------------------------------------------------------------------
# Source registry + mixing
# ---------------------------------------------------------------------------


SourceName = Literal["binary-30k", "local-dir"]


def make_source(name: str, **kwargs) -> Source:
    """Construct a Source by name. Used by CLI.

    ``local-dir`` requires a ``root=`` keyword (the directory to walk).
    """
    if name == "binary-30k":
        return Binary30kSource()
    if name == "local-dir":
        if "root" not in kwargs:
            raise ValueError("local-dir source requires root=<directory>")
        return LocalDirectorySource(**kwargs)
    raise ValueError(
        f"unknown source {name!r}; expected one of binary-30k / local-dir"
    )


def mix_records(
    sources: list[Source], split: Split,
    per_source_limit: dict[str, int] | None = None,
    require_exists: bool = True,
) -> list[TrainingRecord]:
    """Combine records from multiple sources, deduping by (source, sha256)."""
    seen: set[tuple[str, str]] = set()
    out: list[TrainingRecord] = []
    for src in sources:
        lim = (per_source_limit or {}).get(src.name)
        recs = src.records(split, limit=lim)
        for r in recs:
            key = (r.source, r.sha256)
            if key in seen:
                continue
            seen.add(key)
            if require_exists and not r.exists():
                continue
            out.append(r)
    return out


__all__ = [
    "Binary30kSource", "LocalDirectorySource", "Source", "SourceName",
    "TrainingRecord", "assign_split_by_hash", "make_source", "mix_records",
]
