"""MimeLens training stack: byte vs binary-BPE MLM pretraining for small encoders.

The performance-critical data layer (file sampling, tokenization, MLM masking) lives
in the `binary_embedding._native` Rust extension. This Python package is the
orchestration layer: configs, model, and the training loop.

Reference implementation for the MimeLens paper. See README.md and docs/.
"""

from binary_embedding import _native

__all__ = ["_native"]
__version__ = "0.1.0"
