"""Project-wide constants. Single source of truth for IDs that must agree across pipelines."""

# bbpe convention: special tokens at IDs 0..6 in this exact order.
# We mirror this for the byte pipeline so masking, padding, and CLS pooling are
# byte/BPE-agnostic at the model level.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|start|>",
    "<|end|>",
    "<|pad|>",
    "<|unk|>",
    "<|cls|>",
    "<|sep|>",
    "<|mask|>",
)

START_ID, END_ID, PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = range(7)

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
NUM_BYTES = 256

# Byte pipeline vocab: specials first, then 256 bytes shifted by NUM_SPECIAL_TOKENS.
BYTE_VOCAB_SIZE = NUM_SPECIAL_TOKENS + NUM_BYTES  # 263
BYTE_OFFSET = NUM_SPECIAL_TOKENS  # byte b -> id b + 7

HF_TOKENIZERS = {
    4_096: "mjbommar/binary-tokenizer-001-4k",
    8_192: "mjbommar/binary-tokenizer-001-8k",
    16_384: "mjbommar/binary-tokenizer-001-16k",
    32_768: "mjbommar/binary-tokenizer-001-32k",
    65_536: "mjbommar/binary-tokenizer-001-64k",
}

DATASET_NAME = "mjbommar/binary-30k-tokenized"
