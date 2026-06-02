# 04 — Training Protocol

## 1. Hardware budget

Single workstation:

- 2 × NVIDIA RTX 4060 Ti, 16 GB VRAM each. No NVLink (PCIe DDP).
- Enough disk for ~80 GB of pre-tokenized caches (see `docs/03-data-pipeline.md` §4).

We use DDP across both GPUs. No model parallelism. No CPU offload (single-node
homelab; ZeRO-3 isn't worth the complexity here).

## 2. Per-step shape

| Knob | Value | Why |
|---|---|---|
| Per-GPU batch | 32 sequences | fits comfortably alongside FlashAttention 2 + bf16 |
| Gradient accumulation | 4 | effective batch = 32 × 4 × 2 GPUs = **256** |
| Sequence length | 1024 (headline) | covers the 1 KB / 4 KB header probes for both pipelines (4 KB > 1024 byte tokens, but the *byte* eval truncates to 1024 tokens regardless) |
| Effective tokens / step | 256 × 1024 = 262 144 | |
| Effective **bytes** / step (byte) | 262 144 | |
| Effective **bytes** / step (bpe-64k) | 262 144 × ~2.89 ≈ 757 K | |

## 3. Matched-bytes scheduling

This is the load-bearing protocol decision. The two pipelines are scheduled to
**see the same total bytes**, *not* the same total tokens:

| Pipeline | Bytes/step | Steps for 30 G bytes |
|---|---:|---:|
| `byte` | 262 144 | ~115 K |
| `bpe-4k` (~2.0 B/tok) | ~524 K | ~57 K |
| `bpe-8k` (~2.2 B/tok) | ~577 K | ~52 K |
| `bpe-16k` (~2.4 B/tok) | ~629 K | ~48 K |
| `bpe-32k` (~2.6 B/tok) | ~681 K | ~44 K |
| `bpe-64k` (~2.9 B/tok) | ~757 K | ~40 K |

We pick a target of **~30 G bytes seen** for the headline schedule (≈ 1 epoch of
the 30 GB corpus, in *bytes*). Step counts vary across pipelines so the
byte-budget is identical.

**Why bytes, not tokens or compute:**

- Bytes are the conserved quantity that's actually meaningful to "how much of
  the corpus did the model see".
- Token-matched scheduling rewards the byte pipeline (longer sequences, fewer
  bytes/sequence) — it'd see far less of the corpus per step.
- Compute-matched scheduling is harder to communicate and requires accurate
  FLOP counters across two architectures with different vocab sizes; we
  include it as a secondary ablation, not the headline.

A secondary "match FLOPs" protocol is reported as ablation in the paper.

## 4. Optimizer + schedule

| Knob | Value | Source |
|---|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.98, ε=1e-6, wd=0.01) | RoBERTa / MosaicBERT |
| Peak LR | 5e-4 | 30 M-param scale, MosaicBERT-tuned |
| Warmup | 6 % of total steps, linear | MosaicBERT default |
| Decay | cosine to 10 % of peak | |
| Grad clip | 1.0 | |
| Weight init | scaled Xavier for attention out-proj; truncated normal (std=0.02) elsewhere | |

We **freeze all hyperparameters across pipelines**. The byte and BPE runs
use bit-identical optimizer configs (modulo step count, which falls out of
the byte-budget rule).

## 5. MLM objective details

- **Mask ratio: 30 %.** MosaicBERT showed 30 % beats 15 % for fast training of
  small encoders with modern recipes.
- Per-position decision: 80 % `<|mask|>`, 10 % random in-vocab, 10 % unchanged.
  Standard.
- Mask whole *tokens*, not whole *spans*. Span masking is interesting but adds
  a confounder ("BPE tokens are longer ⇒ span masking covers more bytes").
- Loss is computed only on masked positions, weighted equally.

## 6. Logging and checkpointing

- `wandb` per run. Project: `binary-embedding-paper`. Run name = config slug.
- Metrics: loss, MLM accuracy, **bits/byte** (the canonical cross-vocab metric),
  grad norm, LR, throughput (sequences/s, bytes/s, tokens/s).
- Checkpoint every 5 K steps and at end. Keep last + best (by val loss).
- Save with `safetensors`. Push final checkpoints to HF Hub
  (`mjbommar/binary-embedding-{byte,bpe-4k,bpe-8k,bpe-16k,bpe-32k,bpe-64k}`).

## 7. Wall-clock budget

Rough estimates based on FlashAttention 2 + bf16 throughput on a 4060 Ti:

| Pipeline | Steps | Steps/sec/GPU (est.) | Wall-clock (DDP × 2) |
|---|---:|---:|---:|
| byte | 115 K | ~2.5 | ~6.4 h |
| bpe-4k | 57 K | ~2.5 | ~3.2 h |
| bpe-64k | 40 K | ~2.5 | ~2.2 h |

Across 6 pipelines × 3 seeds ≈ 18 runs ≈ **3 GPU-days**. Comfortable in a
calendar week with breathing room for retries.

These estimates are validated by a 1 % dry run before any full grid kicks off
(`scripts/dry_run.sh`).

## 8. Ablations (secondary, after the headline grid)

Each ablation is a single-axis sweep, headline config otherwise unchanged:

| Ablation | What changes | What it tells us |
|---|---|---|
| **Match FLOPs** | step count s.t. forward FLOPs match across pipelines | does the byte/BPE story flip if we control compute instead of bytes? |
| **Seq 2048** | seq len 1024 → 2048; same bytes seen | is the BPE win sensitive to context? |
| **Mask ratio 15 %** | 30 % → 15 % | sanity: does MosaicBERT's recipe help or hurt the comparison? |
| **Document weighting** | uniform ↔ `sqrt(n_bytes)` sampling | how much do giant binaries matter? |
| **No malware** | drop ~21 % malware from train | is the win driven by malware section noise? |

## 9. What "done" looks like

A pretraining run is *done* when all of the following hold:

- [ ] Hit the target bytes-seen number (no more, no less, ± 1 %).
- [ ] MLM bits/byte at val has been monotonically non-increasing for the last
      3 checkpoints.
- [ ] Probe MLP accuracy on a tiny held-out probe set (200 files, sanity only)
      is non-trivially above chance.
- [ ] `wandb` run is closed and tagged with the config hash.
- [ ] Checkpoint pushed to HF Hub.

If any condition fails, we don't move on to evaluation — we debug the run.

## 10. Reproducibility

- One config file per run (`configs/runs/<slug>.yaml`). Config hash logged.
- Seeds: dataset shuffling, init, dropout, masking, all derived from a single
  `seed` field. Three seeds for headline runs.
- Lockfile: `uv.lock` plus `git rev-parse HEAD` recorded in every wandb run.
- Determinism: `runtime.deterministic: true` flips
  `torch.use_deterministic_algorithms(True)` and sets `CUBLAS_WORKSPACE_CONFIG`
  — for dry runs only (it kills FA/SDPA fast paths). Production training is
  non-deterministic but seeded; we average over 3 seeds.

## 11. Configurability surface

The config schema (pydantic v2, see `python/binary_embedding/training/config.py`)
exposes every hyperparameter the paper can plausibly need. Highlights:

- **`loss`** is a discriminated union: `mlm` / `contrastive` / `classification`
  / `byol` / `composite[parts=…, aux_warmup_steps=N]`. Composite linearly
  combines named parts with weights and ramps non-MLM contributions in over
  `aux_warmup_steps`.
- **`optim.optimizer`** picks `adamw` (default, fused) / `lion` / `adafactor` /
  `adamw8bit`. The latter three load lazily so missing deps fail with a clear
  message.
- **`optim.param_groups`** lets you (a) override the regex that selects
  no-decay parameters, (b) add custom regex-based LR/WD rules
  (`{pattern, lr_multiplier, weight_decay, name}`), and (c) enable LLRD with a
  single decay factor.
- **`schedule`** is one of `cosine` / `linear` / `wsd` / `constant`, with
  warmup expressible as `warmup_steps` / `warmup_pct` / `warmup_bytes`
  (mutually exclusive).
- **`schedule_budget`** is the total-amount knob: exactly one of
  `target_bytes` / `target_steps` / `target_tokens`.
- **`reg`** controls `embedding_dropout`, `hidden_dropout`,
  `attention_dropout`, `drop_path_rate` (linearly scheduled across depth),
  `layer_scale_init`, `init_scheme` ∈ {trunc_normal, xavier, scaled_residual},
  `ema_decay` (None disables).
- **`runtime`** controls `precision` (fp32/bf16/fp16), `matmul_precision`,
  `compile_mode` (none/default/reduce-overhead/max-autotune), `deterministic`,
  `cudnn_benchmark`.
- **`log`** controls `out_dir`, `log_every`, `save_every`, `eval_every`,
  `eval_files`, `track_best_metric`, `track_best_mode`, `keep_last_k`,
  `wandb_project`, `wandb_run_name`, plus a list of extra `callbacks`.

**CLI overrides** without editing YAML:

```bash
uv run python -m binary_embedding.training.pretrain \
    --config configs/runs/headline_byte.yaml \
    -O optim.lr=3e-4 \
    -O schedule_budget.target_bytes=10_000_000_000 \
    -O loss.parts.0.weight=0.7
```

Each override is parsed as YAML so types come out right (`true` → bool,
`null` → `None`, `[1,2]` → list).

## 12. Run artefacts

Every successful run leaves the following under `cfg.log.out_dir`:

- `config.resolved.yaml` — the merged + validated config with `config_hash`
  and `git_sha` in the leading comment.
- `manifest.json` — config hash, git SHA + dirty bit, GPU info, Python/torch/
  CUDA versions, uv.lock SHA, wandb run id.
- `events.jsonl` — append-only log of `start`, `step`, `eval`, `checkpoint`,
  `end` events. Used by the eval/notebook tier.
- `train.log` — human-readable console log.
- `checkpoints/state.pt` — full state for resume (model, optimizer, EMA, RNG).
- `checkpoints/step_NNNNNNNN.safetensors` — periodic weights-only saves
  (last `keep_last_k`).
- `checkpoints/best.safetensors` — best-by-metric (`track_best_metric`).
