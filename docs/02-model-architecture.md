# 02 — Model Architecture

## 1. Goal

Train a *small*, *modern*, *encoder-only* transformer that fits in workstation
budget and is rich enough to test the input-pipeline hypotheses. Two variants
that differ **only in their embedding table and tokenizer**.

## 2. Headline configuration

| Component | Choice | Why |
|---|---|---|
| Family | encoder-only, MLM-pretrained | matches the BERT lineage and the eval tasks |
| Layers | 8 | a respectable depth at small width; fits 16 GB |
| Hidden size (`d_model`) | 384 | divisible by every head count we'd plausibly try |
| FFN multiplier | 8/3 (GeGLU) | matches LLaMA-style; ≈ d_model × 1024 effective |
| Attention heads | 6 (head dim 64) | head dim 64 is the FlashAttention sweet spot |
| Attention | RoPE positions, FlashAttention 2 | linear in seq len; no positional embedding table |
| Normalization | RMSNorm, pre-norm | MosaicBERT-validated, bf16-stable |
| Activation | GeGLU | ModernBERT default; small loss improvement over GELU |
| Dropout | 0.0 (pretrain), 0.1 (probe head) | MLM prefers no dropout under modern recipes |
| Padding | unpadded (variable-length) | MosaicBERT-style; ≈ 30 % wall-clock saving |
| Precision | bf16 weights + grads, fp32 master | RTX 4060 Ti has good bf16 throughput |
| Sequence length | 1024 (headline), 2048 (ablation) | covers 1 KB / 4 KB / 16 KB header probes for *both* variants |

Parameter counts (excluding embedding table):

| Component | Params |
|---|---|
| 8 × attention | ~3.5 M |
| 8 × GeGLU FFN | ~12 M |
| LayerNorms / biases | ~0.05 M |
| **Backbone total** | **~15.5 M** |

Embedding table:

| Variant | Vocab | Embedding params (vocab × 384) | Tied? |
|---|---:|---:|---|
| `byte` | 263 (256 + 7 specials) | ~0.10 M | input ↔ MLM head |
| `bpe-4k` | 4 096 | 1.57 M | tied |
| `bpe-8k` | 8 192 | 3.15 M | tied |
| `bpe-16k` | 16 384 | 6.29 M | tied |
| `bpe-32k` | 32 768 | 12.58 M | tied |
| `bpe-64k` | 65 536 | 25.16 M | tied |

**Trainable param totals:** byte ≈ 15.6 M; bpe-4k ≈ 17 M; bpe-64k ≈ 41 M. We
report both backbone-only and full-model param counts. The MLM head shares the
embedding matrix, so the only "free" parameters from a larger vocab are the
embedding lookup itself.

## 3. Heads and pooling

- **MLM head**: projects hidden → vocab, weight-tied to the input embedding.
  Trained with cross-entropy over masked positions.
- **CLS pooler**: a learned linear projection of the position-0 hidden state to
  a 256-dim L2-normalized vector. Used in clustering and as the input to all
  linear probes.
- **Probe heads** (eval only): single linear layer; no fine-tuning of the
  encoder body. Optionally a 2-layer MLP probe as a sanity check.

## 4. What we *deliberately don't* do

- **No NSP / sentence order.** Single MLM objective. Cleaner attribution, less
  noise.
- **No alternating local/global attention** (despite ModernBERT). At seq 1024 the
  speedup is small and the implementation surface is large. We can revisit if
  we extend to 8K context.
- **No RetNet / SSM.** Out of scope; muddies the comparison.
- **No tokenizer-aware tying tricks.** Both variants share the same MLM head
  shape rules (tied to input embedding); we don't try to "help" the byte model
  with auxiliary losses.

## 5. Special tokens

`bbpe`-trained tokenizers reserve IDs 0–6 for `<|start|>`, `<|end|>`, `<|pad|>`,
`<|unk|>`, `<|cls|>`, `<|sep|>`, `<|mask|>` (see sibling-paper conventions).
For the byte model we adopt the same convention: IDs 0–6 reserved as specials,
then bytes occupy IDs 7–262 (so `byte_id(b) = b + 7`). This keeps:

- `<|cls|>` always at position 0 of every sequence.
- A common `pad_id`, `mask_id`, `cls_id` constant across both variants.
- The masking implementation identical regardless of pipeline.

## 6. How the two variants are kept matched

| Knob | byte | bpe-Nk | matched? |
|---|---|---|---|
| Backbone params | identical | identical | ✅ |
| Layers / d_model / heads | identical | identical | ✅ |
| Optimizer + LR + schedule | identical | identical | ✅ |
| Total **bytes** of training data | identical | identical | ✅ |
| Total **tokens** of training data | (≈ 3× larger) | (smaller) | ❌ — by design |
| Total compute (FLOPs) | varies with seq len | varies with seq len | ⚠ — see §7 |
| Sequence length | 1024 | 1024 | ✅ |
| MLM mask ratio | 30 % | 30 % | ✅ |

## 7. The compute-matching subtlety

At fixed sequence length, the BPE encoder sees *more bytes per forward* than the
byte encoder. So either:

- **Match steps** → BPE sees more bytes → BPE has an unfair advantage.
- **Match bytes** → BPE runs fewer steps → byte has more updates per byte.
- **Match FLOPs** → both sides at different step counts but same compute.

We adopt **"match bytes seen"** as the headline protocol (cleaner story, easier
to communicate) and **"match FLOPs"** as a secondary ablation. Details in
`docs/04-training-protocol.md` §3.

## 8. Implementation choices

- Base on `transformers.PreTrainedModel` so we get `from_pretrained` + safetensors
  for free.
- Borrow the FlashAttention 2 + RoPE block implementation from `modeling_modernbert`
  rather than rolling our own.
- Custom embedding-table init: byte model uses unit-norm Gaussian; BPE model
  uses HF default (truncated normal, std=0.02). Documented and seeded.

## 9. Open architectural questions

- Whether to share the **CLS pooler** weights across variants for a more direct
  embedding comparison. Default: separate (each variant gets to learn its own
  pool). Add a "shared-pool" ablation if numbers look suspicious.
- Whether to add a small **byte-grouping conv** (CANINE-style) for the byte
  variant. Resists the temptation in v1: confounds the comparison. Worth a
  follow-up paper.
