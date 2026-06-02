"""Load the binary-30k stratified splits and resolve file paths to local bytes.

The corpus ships as arrow shards (one subdirectory per split) whose metadata
schema includes a ``file_path`` column pointing at the file bytes on disk. Set
``MIMELENS_BINARY30K_SPLITS`` to the directory that contains the ``train`` /
``validation`` / ``test`` subdirectories of arrow shards.

If your shards were written on a host where the bytes lived at a different mount
than they do now, set ``MIMELENS_PATH_REMAP="<old-prefix>::<new-prefix>"`` and
``file_path`` values are rewritten on load. Identity when unset.
"""

from __future__ import annotations

import dataclasses as dc
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import pyarrow as pa

DEFAULT_SPLITS_ROOT = Path(
    os.environ.get("MIMELENS_BINARY30K_SPLITS", "data/binary-30k/splits")
)

Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "test")


def _remap_pair() -> tuple[str, str] | None:
    raw = os.environ.get("MIMELENS_PATH_REMAP", "")
    if "::" not in raw:
        return None
    old, new = raw.split("::", 1)
    return (old, new) if old else None


def rewrite_path(stale: str) -> str:
    """Optionally remap a stored file path to where the bytes live now.

    Controlled by ``MIMELENS_PATH_REMAP="<old-prefix>::<new-prefix>"``; identity
    when unset.
    """
    pair = _remap_pair()
    if pair and stale.startswith(pair[0]):
        return pair[1] + stale[len(pair[0]) :]
    return stale


@dc.dataclass(frozen=True, slots=True)
class Record:
    """A single binary in the corpus.

    Carries only the fields needed for cache building + slicing. We deliberately
    drop the heavy `tokens` column (it's recomputed per variant) and keep only
    the metadata that ends up in the index parquet.
    """

    sha256: str
    file_path: str
    file_size: int
    platform: str
    file_format: str
    architecture: str
    binary_type: str
    is_malware: bool
    is_packed: bool
    is_signed: bool
    is_stripped: bool
    entropy: float
    split: Split

    def exists(self) -> bool:
        return Path(self.file_path).is_file()


_PROJECTION_COLUMNS: tuple[str, ...] = (
    "sha256",
    "file_path",
    "file_size",
    "platform",
    "file_format",
    "architecture",
    "binary_type",
    "is_malware",
    "is_packed",
    "is_signed",
    "is_stripped",
    "entropy",
)


def _read_split(split_root: Path, split: Split) -> pa.Table:
    """Read one split's arrow shards as a pyarrow Table, projecting metadata cols only."""
    split_dir = split_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split directory: {split_dir}")
    shards = sorted(split_dir.glob("data-*.arrow"))
    if not shards:
        raise FileNotFoundError(f"no arrow shards in {split_dir}")
    tables: list[pa.Table] = []
    for shard in shards:
        with pa.ipc.open_stream(shard) as reader:
            tables.append(reader.read_all().select(list(_PROJECTION_COLUMNS)))
    return pa.concat_tables(tables)


def load_split_table(split: Split, root: Path | None = None) -> pa.Table:
    """Return one split as a pyarrow Table with file paths already rewritten."""
    root = root or DEFAULT_SPLITS_ROOT
    table = _read_split(root, split)
    paths = pa.array([rewrite_path(p) for p in table.column("file_path").to_pylist()])
    return table.set_column(table.schema.get_field_index("file_path"), "file_path", paths)


def load_records(
    split: Split,
    root: Path | None = None,
    *,
    require_exists: bool = False,
    limit: int | None = None,
    shuffle_seed: int | None = None,
) -> list[Record]:
    """Materialize a split's records as a list of `Record`s.

    - `require_exists=True` filters to files that resolve on the local filesystem.
    - `shuffle_seed` shuffles before applying `limit`. The on-disk arrow shards
      are class-clumped, so taking the first N rows yields one file_format /
      platform — set a seed any time you're going to slice a small subset.
    """
    table = load_split_table(split, root)
    cols = {name: table.column(name).to_pylist() for name in _PROJECTION_COLUMNS}
    n = table.num_rows
    order = list(range(n))
    if shuffle_seed is not None:
        import random as _random

        _random.Random(shuffle_seed).shuffle(order)
    out: list[Record] = []
    for i in order:
        rec = Record(
            sha256=cols["sha256"][i],
            file_path=cols["file_path"][i],
            file_size=int(cols["file_size"][i]),
            platform=cols["platform"][i],
            file_format=cols["file_format"][i],
            architecture=cols["architecture"][i],
            binary_type=cols["binary_type"][i],
            is_malware=bool(cols["is_malware"][i]),
            is_packed=bool(cols["is_packed"][i]),
            is_signed=bool(cols["is_signed"][i]),
            is_stripped=bool(cols["is_stripped"][i]),
            entropy=float(cols["entropy"][i]),
            split=split,
        )
        if require_exists and not rec.exists():
            continue
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


def split_of(sha256: str, root: Path | None = None) -> Split | None:
    """Return the split that owns `sha256`, or None if not in the corpus.

    Builds the index lazily on first call (cached for the lifetime of the
    process). For one-off lookups; for bulk work, use `build_split_index`.
    """
    return _split_index_cached(root)._index.get(sha256)


@dc.dataclass(frozen=True, slots=True)
class SplitIndex:
    _index: dict[str, Split]

    def assert_split(self, sha256: str, expected: Split) -> None:
        got = self._index.get(sha256)
        if got != expected:
            raise AssertionError(
                f"sha256 {sha256[:12]}.. is in split={got!r}, expected {expected!r}"
            )

    def assert_no_leak(
        self,
        train_sha: Iterable[str],
        eval_sha: Iterable[str],
    ) -> None:
        train_set = set(train_sha)
        for s in eval_sha:
            if s in train_set:
                raise AssertionError(f"sha256 {s[:12]}.. appears in both train and eval")

    def __len__(self) -> int:
        return len(self._index)

    def items(self) -> Iterator[tuple[str, Split]]:
        return iter(self._index.items())


def build_split_index(root: Path | None = None) -> SplitIndex:
    """Build a sha256 -> split mapping spanning all three splits."""
    out: dict[str, Split] = {}
    for split in SPLITS:
        table = load_split_table(split, root)
        for sha in table.column("sha256").to_pylist():
            if sha in out and out[sha] != split:
                raise AssertionError(
                    f"sha256 {sha[:12]}.. claimed by both {out[sha]!r} and {split!r}"
                )
            out[sha] = split
    return SplitIndex(_index=out)


_CACHED_INDEX: SplitIndex | None = None


def _split_index_cached(root: Path | None) -> SplitIndex:
    # Process-wide cache. Tests can clear via `clear_split_index_cache`.
    global _CACHED_INDEX
    if _CACHED_INDEX is None:
        _CACHED_INDEX = build_split_index(root)
    return _CACHED_INDEX


def clear_split_index_cache() -> None:
    global _CACHED_INDEX
    _CACHED_INDEX = None
