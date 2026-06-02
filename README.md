# mimelens-training

Reference training stack for **MimeLens** — a family of small BERT-style
encoders that classify file content type (libmagic MIME labels) from a byte
chunk taken at *any* offset inside a file, not just the head. MimeLens is
pretrained MLM-only on windows sampled at uniformly random offsets, so one
checkpoint serves streaming, fragment, packet-payload, random-seek, and
header-corrupted inputs that whole-file classifiers are not built for.

This repository is the **training code as a reference implementation**: the
model, the MLM pretraining loop, the input pipelines (raw bytes and binary
BPE), and the packed-token data layer. It is meant to document and reproduce the
*method*. It is not a turnkey rerun of the paper's exact checkpoints — the full
33 GB pretraining corpus is mixed-provenance and is not redistributable (see
[Data](#data)).

- **Models:** released on the Hugging Face Hub under
  [`mjbommar/mimelens-001-*`](https://huggingface.co/mjbommar).
- **Tokenizers:** [`mjbommar/binary-tokenizer-001-*`](https://huggingface.co/mjbommar)
  (consumed via the [`bbpe`](https://github.com/mjbommar/binary-bpe) crate).
- **Public anchor corpus:** [`mjbommar/binary-30k-tokenized`](https://huggingface.co/datasets/mjbommar/binary-30k-tokenized).

## What's here

```
mimelens-training/
├── python/binary_embedding/
│   ├── constants.py          # shared ids (specials, byte vocab), HF names
│   ├── models/               # the BinaryEncoder (RoPE, RMSNorm, GeGLU, mean-pool)
│   ├── data/                 # packed-token cache, dataset, splits, sources
│   ├── training/             # config, trainer, losses, optim, schedules, callbacks
│   └── analysis/             # small embedding/analysis helpers
├── src/                      # Rust _native extension (sampling, BPE tokenize, MLM mask)
├── configs/runs/             # YAML run configs (model + tokenizer + schedule)
├── scripts/build_cache.py    # pre-encode a corpus to a packed token cache
├── docs/                     # model architecture + training protocol
└── tests/                    # unit tests for the training stack and native module
```

Hot-path data work (file sampling, BPE tokenization, MLM masking) is Rust under
PyO3 (`binary_embedding._native`); everything else is Python. The Python package
keeps the historical import name `binary_embedding`.

## Install

Requires Python 3.12+ and a Rust toolchain (the native extension builds with
[maturin](https://www.maturin.rs/)).

```bash
# with uv (recommended)
uv sync                       # builds the native extension and installs deps
# or, for a fast iterate-on-Rust loop
uv run maturin develop --release
```

The native extension depends on the published [`bbpe`](https://github.com/mjbommar/binary-bpe)
crate (see `Cargo.toml`). Optional extra optimizers (bitsandbytes, lion) install
with `pip install mimelens-training[optimizers]`.

## Data

Two data sources ship in `binary_embedding.data.sources`:

- **`Binary30kSource`** — the public `binary-30k` corpus (29,793 ELF / PE /
  Mach-O / APK binaries) with stratified 70/15/15 splits. Point the loader at
  your local copy of the arrow shards with `MIMELENS_BINARY30K_SPLITS`. If the
  shards record paths from another mount, remap with
  `MIMELENS_PATH_REMAP="<old-prefix>::<new-prefix>"`.
- **`LocalDirectorySource`** — bring-your-own corpus. Walks a directory tree,
  emits one record per file, and assigns deterministic hash-bucketed splits.
  This is the extension point for a larger mixed corpus.

The binary-BPE tokenizer JSONs are resolved from `MIMELENS_TOKENIZER_DIR`
(default `./tokenizers`) in the layout `tokenizer-{4k,8k,16k,32k,64k}/...`, or
download them from the Hugging Face Hub (`mjbommar/binary-tokenizer-001-*`).

> **Not included.** The paper's full 33 GB pretraining corpus (magic-corpus
> extracts, packed binaries, and Windows drivers added to binary-30k) and the
> evaluation corpora are not redistributable and are not part of this repository.
> The released checkpoints and this code are the reproducible surface.

## Quickstart

```bash
# 1. Build a packed token cache for one (variant, split).
python scripts/build_cache.py --variant byte --source binary-30k --split train
python scripts/build_cache.py --variant byte --source binary-30k --split validation

# 2. Pretrain (MLM-only). Configs live in configs/runs/.
python -m binary_embedding.training.pretrain --config configs/runs/headline_byte.yaml

# Override any config field on the CLI:
python -m binary_embedding.training.pretrain \
    --config configs/runs/headline_bpe16k.yaml --override optim.lr=3e-4
```

Each run writes checkpoints, a JSONL metrics log, and a `manifest.json`
(config hash, git SHA, GPU info, env) under its `log.out_dir`. Training
auto-resumes from `out_dir/checkpoints/state.pt`.

## Using the released models

You don't need this training package to *run* MimeLens — the released cells load
straight from the Hugging Face Hub via `transformers`:

```bash
pip install transformers torch
python examples/classify.py path/to/file
```

See [`examples/classify.py`](examples/classify.py) for the input contract and
window-selection guidance (the model reads the first ~1,022 tokens of whatever
you pass; a short head window classifies magic-byte / compressed types better
than a long one). The deployed cells (`mjbommar/mimelens-001-medium-{bpe-16k,byte,bpe-64k}-s1`)
ship a baked 125-class classifier head; the rest expose the mean-pooled encoder.

## The model family

A single architecture (pre-norm transformer; RoPE θ=10000, RMSNorm, GeGLU 8/3,
tied embeddings, mean-pooled body tokens) trained across three axes:

| axis | values |
|---|---|
| size | tiny (~3.2M), small (~14M), medium (~38M) backbone params |
| input pipeline | raw bytes, plus binary BPE at 4k / 8k / 16k / 32k / 64k vocab |
| context length | 1024 tokens (main), 256 tokens (short-context cells) |

Matched-compute discipline holds architecture, optimizer, schedule, and
total-bytes-seen constant across pipelines; only tokens-seen and vocabulary
differ. See `docs/04-training-protocol.md`.

## Tests

```bash
uv run pytest          # unit tests for the training stack + native module
```

The shipped tests are self-contained (no corpus, no network).

## Citation

If you use this code or the MimeLens models, please cite the MimeLens paper
(see the model cards at https://huggingface.co/mjbommar for the current
reference) and the binary-BPE tokenizers:

```bibtex
@software{bommarito_mimelens_training,
  author = {Bommarito, Michael J.},
  title  = {mimelens-training: reference training stack for MimeLens},
  url    = {https://github.com/mjbommar/mimelens-training},
  year   = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
