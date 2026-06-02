"""CLI entrypoint: `uv run python -m binary_embedding.training.pretrain --config X.yaml`.

Thin shim over `Trainer.train(cfg)`. All semantics live in `trainer.py`,
`losses.py`, `optim.py`, `callbacks.py`, etc.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from binary_embedding.training.config import load_config
from binary_embedding.training.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser(description="binary-embedding pretraining")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--override", "-O", action="append", default=[],
        help="key.path=value overrides (repeatable)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config, overrides=args.override)
    train(cfg)


if __name__ == "__main__":
    main()
