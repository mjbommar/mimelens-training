"""Classify file content type with a released MimeLens model.

Loads a deployed MimeLens cell from the Hugging Face Hub and labels a byte
window with one of libmagic's 125 MIME types. This uses the *released* models
via `transformers` (it does not need this training package installed):

    pip install transformers torch
    python examples/classify.py path/to/file [more files ...]

Choosing a window
-----------------
The model reads the first ~1,022 tokens of whatever you pass — a *prefix* of the
buffer, not the whole window. Two practical consequences:

- For magic-byte-bearing or compressed types (PNG, ZIP, GZIP, JPEG), a short
  head window (256 B-1 KB) classifies better than 4 KB: a long high-entropy body
  dilutes the header signal within the fixed token budget, and the model returns
  `application/octet-stream` on a mostly-opaque window (correct behaviour for
  genuinely high-entropy input).
- For fragments / packets you cannot choose the offset — pass what you have.
  That is the regime MimeLens is built for.
"""

from __future__ import annotations

import argparse

# The clean-head classifier cell (ships a baked 125-class head, so the
# text-classification pipeline works out of the box). For streaming / packet
# inputs, mjbommar/mimelens-001-medium-byte-s1 is the recommended cell.
DEFAULT_MODEL = "mjbommar/mimelens-001-medium-bpe-16k-s1"


def classify(paths: list[str], *, model_id: str, window: int, top_k: int) -> None:
    from transformers import pipeline

    clf = pipeline(
        "text-classification",
        model=model_id,
        trust_remote_code=True,  # the cell ships custom modeling code
        top_k=top_k,
    )
    for path in paths:
        with open(path, "rb") as f:
            buf = f.read(window)
        # latin-1 is a bijection over bytes 0-255, matching how the tokenizers
        # were trained. The pipeline truncates to the model's token budget.
        preds = clf(buf.decode("latin-1"))[0]
        top = ", ".join(f"{p['label']} ({p['score']:.3f})" for p in preds)
        print(f"{path}\t{top}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", help="file(s) to classify")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--window", type=int, default=1024,
                    help="bytes to read from the file head (default 1024; use a "
                         "shorter window for magic-byte / compressed types)")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    classify(args.paths, model_id=args.model, window=args.window, top_k=args.top_k)


if __name__ == "__main__":
    main()
