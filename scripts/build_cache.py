"""Build a packed token cache for one (variant, split) from a data source.

Pre-encodes a corpus to a packed ``tokens.bin`` + sidecar ``index.parquet`` so
training never re-tokenizes. See `binary_embedding.data.cache` for the layout
and `binary_embedding.data.sources` for the available sources.

Examples
--------
Build the byte-variant train cache from the public binary-30k corpus::

    python scripts/build_cache.py --variant byte --source binary-30k --split train

Build a bpe-16k cache from a local directory of files you hold yourself::

    python scripts/build_cache.py --variant bpe-16k --source local-dir \\
        --root /path/to/corpus --split train --cache-root data/cache
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from binary_embedding.constants import HF_TOKENIZERS
from binary_embedding.data import cache as cache_mod
from binary_embedding.data.sources import Binary30kSource, LocalDirectorySource
from binary_embedding.data.splits import SPLITS

# variant name -> bpe vocab size (None for the byte variant)
_BPE_VOCAB = {f"bpe-{v // 1024}k": v for v in sorted(HF_TOKENIZERS)}
VARIANTS = ["byte", *_BPE_VOCAB]


def _make_variant(name: str) -> cache_mod.Variant:
    if name == "byte":
        return cache_mod.variant_byte()
    if name in _BPE_VOCAB:
        return cache_mod.variant_bpe(_BPE_VOCAB[name])
    raise SystemExit(f"unknown variant {name!r}; expected one of {VARIANTS}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--source", default="binary-30k", choices=["binary-30k", "local-dir"])
    p.add_argument("--root", type=Path, help="directory to walk (required for --source local-dir)")
    p.add_argument("--split", default="train", choices=[*SPLITS, "all"])
    p.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    p.add_argument("--limit", type=int, default=None, help="cap files per split (debugging)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.source == "local-dir":
        if args.root is None:
            raise SystemExit("--source local-dir requires --root")
        source = LocalDirectorySource(root=args.root)
    else:
        source = Binary30kSource()

    variant = _make_variant(args.variant)
    splits = SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        per_source_limit = {source.name: args.limit} if args.limit is not None else None
        manifest = cache_mod.build_mixed_cache(
            variant=variant,
            split=split,
            cache_root=args.cache_root,
            sources=[source],
            per_source_limit=per_source_limit,
        )
        logging.info(
            "built %s/%s: %d files, %d tokens, %.2f GB on disk",
            manifest.variant, manifest.split, manifest.n_files, manifest.n_tokens,
            manifest.on_disk_bytes / 1e9,
        )


if __name__ == "__main__":
    main()
