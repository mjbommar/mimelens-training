"""Pydantic v2 configuration models for the training stack.

Hierarchy:

    RunConfig
    ├── ModelCfg
    ├── DataCfg
    ├── OptimCfg          # optimizer + param groups
    ├── ScheduleCfg       # lr schedule
    ├── LossCfg           # discriminated union: mlm | composite | ...
    ├── RegCfg            # ema, dropout, drop-path, init
    ├── RuntimeCfg        # precision, compile, determinism, ckpt
    ├── LogCfg            # callbacks, cadence, out_dir
    └── BatchingCfg       # per-gpu batch, grad accum, num_workers

Loaded from YAML, optionally augmented by `--override key.path=value` pairs,
fully validated by pydantic, then dumped (with all defaults filled in) to
`<out_dir>/config.resolved.yaml`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


# ---------------------------------------------------------------------------
# Variant pipeline
# ---------------------------------------------------------------------------


class VariantCfg(_StrictModel):
    """Which input pipeline this run uses."""

    pipeline: Literal[
        "byte", "bpe-4k", "bpe-8k", "bpe-16k", "bpe-32k", "bpe-64k"
    ] = "byte"


# ---------------------------------------------------------------------------
# Model + regularization
# ---------------------------------------------------------------------------


InitScheme = Literal["trunc_normal", "xavier", "scaled_residual"]


class RegCfg(_StrictModel):
    embedding_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0
    layer_scale_init: float | None = None  # None = disabled, else e.g. 1e-4
    init_scheme: InitScheme = "trunc_normal"
    init_std: float = 0.02
    ema_decay: float | None = None  # None disables EMA; else e.g. 0.999

    @field_validator(
        "embedding_dropout", "hidden_dropout", "attention_dropout", "drop_path_rate"
    )
    @classmethod
    def _in_unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be in [0, 1]")
        return v

    @field_validator("ema_decay")
    @classmethod
    def _ema_in_range(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 < v < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        return v


class ModelCfg(_StrictModel):
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 6
    ffn_multiplier_num: int = 8
    ffn_multiplier_den: int = 3
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    cls_pool_dim: int = 256
    grad_checkpointing: bool = False

    @model_validator(mode="after")
    def _check_dims(self) -> "ModelCfg":
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class DataCfg(_StrictModel):
    cache_root: Path = Field(default=Path("data/cache"))
    seq_len: int = 1024
    mask_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    doc_sampling: Literal["uniform", "sqrt_bytes"] = "uniform"
    train_split: Literal["train", "validation", "test"] = "train"
    eval_split: Literal["train", "validation", "test"] = "validation"

    @field_validator("seq_len")
    @classmethod
    def _seq_min(cls, v: int) -> int:
        if v < 4:
            raise ValueError("seq_len must be ≥ 4 (CLS + ≥2 body + SEP)")
        return v


# ---------------------------------------------------------------------------
# Param groups + optimizer + schedule
# ---------------------------------------------------------------------------


class ParamGroupRule(_StrictModel):
    """Override LR/WD for parameters whose qualified name matches `pattern`.

    `pattern` is a regex matched against `model.named_parameters()` names.
    `lr_multiplier` scales the base LR; `weight_decay` overrides the default.
    """

    pattern: str
    lr_multiplier: float = 1.0
    weight_decay: float | None = None  # None = inherit base
    name: str = "custom"


class ParamGroupsCfg(_StrictModel):
    """Default rules: WD=0 for biases/norms, plus user-supplied LR/WD overrides.

    The default `no_decay_pattern` matches:
    - any qualified name ending in `bias` (bare biases anywhere)
    - any name ending in `norm.weight` or `norm{digit}.weight` (RMSNorm, LayerNorm)
    - any name ending in `.gamma` (LayerScale)
    """

    no_decay_pattern: str = r"(\.bias$|norm\d*\.weight$|\.gamma$)"
    custom_rules: list[ParamGroupRule] = Field(default_factory=list)
    llrd_decay: float | None = None  # e.g. 0.9 -> deeper layers get smaller LR
    llrd_layer_pattern: str = r"layers\.(\d+)\."  # capture group = layer index


class AdamWArgs(_StrictModel):
    type: Literal["adamw"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    fused: bool = True


class LionArgs(_StrictModel):
    type: Literal["lion"] = "lion"
    betas: tuple[float, float] = (0.9, 0.99)


class AdafactorArgs(_StrictModel):
    type: Literal["adafactor"] = "adafactor"
    relative_step: bool = False
    scale_parameter: bool = True


class AdamW8bitArgs(_StrictModel):
    type: Literal["adamw8bit"] = "adamw8bit"
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6


OptimizerSpec = Annotated[
    Union[AdamWArgs, LionArgs, AdafactorArgs, AdamW8bitArgs],
    Field(discriminator="type"),
]


class GradClipCfg(_StrictModel):
    by_norm: float | None = 1.0  # None disables
    by_value: float | None = None
    skip_on_nan: bool = True


class OptimCfg(_StrictModel):
    lr: float = 5e-4
    weight_decay: float = 0.01
    optimizer: OptimizerSpec = Field(default_factory=AdamWArgs)
    param_groups: ParamGroupsCfg = Field(default_factory=ParamGroupsCfg)
    grad_clip: GradClipCfg = Field(default_factory=GradClipCfg)


class ScheduleCfg(_StrictModel):
    type: Literal["cosine", "linear", "wsd", "constant"] = "cosine"
    warmup_pct: float | None = 0.06  # alternative to warmup_steps / warmup_bytes
    warmup_steps: int | None = None
    warmup_bytes: int | None = None
    min_lr_pct: float = 0.10
    stable_pct: float | None = None  # for WSD: fraction in stable phase

    @model_validator(mode="after")
    def _exactly_one_warmup(self) -> "ScheduleCfg":
        present = [
            v is not None
            for v in (self.warmup_pct, self.warmup_steps, self.warmup_bytes)
        ]
        if sum(present) != 1:
            raise ValueError(
                "exactly one of warmup_pct / warmup_steps / warmup_bytes must be set"
            )
        return self


# ---------------------------------------------------------------------------
# Losses (discriminated union)
# ---------------------------------------------------------------------------


class MLMLossCfg(_StrictModel):
    type: Literal["mlm"] = "mlm"
    weight: float = 1.0
    name: str = "mlm"


class ContrastiveLossCfg(_StrictModel):
    """SimCSE-style in-batch contrastive on the CLS embedding.

    Two views per file are generated by sampling two random windows; the loss
    pulls views from the same file together, pushes others apart.
    """

    type: Literal["contrastive"] = "contrastive"
    weight: float = 1.0
    name: str = "contrastive"
    temperature: float = 0.05
    proj_dim: int = 256
    proj_hidden_dim: int = 512
    proj_layers: int = 2  # 1 = linear; 2+ = MLP


class ClassificationLossCfg(_StrictModel):
    """Auxiliary cross-entropy head over a metadata column (e.g. file_format)."""

    type: Literal["classification"] = "classification"
    weight: float = 0.1
    name: str = "classification"
    target_column: str = "file_format"
    n_classes: int = 5
    label_smoothing: float = 0.0


class BYOLLossCfg(_StrictModel):
    """BYOL-style projection + EMA-target prediction."""

    type: Literal["byol"] = "byol"
    weight: float = 1.0
    name: str = "byol"
    proj_dim: int = 256
    proj_hidden_dim: int = 512
    target_decay: float = 0.996


SingleLoss = Annotated[
    Union[MLMLossCfg, ContrastiveLossCfg, ClassificationLossCfg, BYOLLossCfg],
    Field(discriminator="type"),
]


class CompositeLossCfg(_StrictModel):
    type: Literal["composite"] = "composite"
    parts: list[SingleLoss]
    grad_norm_balance: bool = False  # scale aux weights inversely by their loss magnitude
    grad_norm_balance_ema: float = 0.99  # EMA decay for the running magnitudes
    grad_norm_balance_clip: float = 10.0  # cap per-part rescale factor
    aux_warmup_steps: int = 0  # step-based linear ramp of non-MLM parts
    aux_warmup_metric: str | None = None  # if set, ramp gated on this scalar metric
    aux_warmup_threshold: float | None = None  # ramp engages when metric < threshold

    @model_validator(mode="after")
    def _gating_consistency(self) -> "CompositeLossCfg":
        if (self.aux_warmup_metric is None) != (self.aux_warmup_threshold is None):
            raise ValueError(
                "aux_warmup_metric and aux_warmup_threshold must be set together"
            )
        return self


LossSpec = Annotated[
    Union[MLMLossCfg, ContrastiveLossCfg, ClassificationLossCfg, BYOLLossCfg, CompositeLossCfg],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Runtime + logging + batching
# ---------------------------------------------------------------------------


class RuntimeCfg(_StrictModel):
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = "none"
    deterministic: bool = False
    cudnn_benchmark: bool = True


class CallbackCfg(_StrictModel):
    type: str
    args: dict[str, Any] = Field(default_factory=dict)


class LogCfg(_StrictModel):
    out_dir: Path = Path("runs/default")
    log_every: int = 50
    save_every: int = 5000  # 0 disables periodic save
    eval_every: int = 1000  # 0 disables in-loop eval
    eval_files: int = 200
    eval_target_label: str = "file_format"
    enable_probe: bool = False  # in-loop logistic-regression probe at eval cadence
    keep_last_k: int = 3  # number of step-* checkpoints to keep
    track_best_metric: str = "eval/mlm_bits_per_byte"
    track_best_mode: Literal["min", "max"] = "min"
    callbacks: list[CallbackCfg] = Field(default_factory=list)
    wandb_project: str | None = None
    wandb_run_name: str | None = None


class BatchingCfg(_StrictModel):
    per_gpu_batch: int = 32
    grad_accum: int = 4
    num_workers: int = 2


# ---------------------------------------------------------------------------
# Schedule budget
# ---------------------------------------------------------------------------


class ScheduleBudgetCfg(_StrictModel):
    """How many bytes (or steps or tokens) to train for. Exactly one must be set."""

    match_axis: Literal["bytes_seen", "steps", "tokens_seen"] = "bytes_seen"
    target_bytes: int | None = None
    target_steps: int | None = None
    target_tokens: int | None = None

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "ScheduleBudgetCfg":
        targets = (self.target_bytes, self.target_steps, self.target_tokens)
        if sum(t is not None for t in targets) != 1:
            raise ValueError(
                "exactly one of target_bytes / target_steps / target_tokens must be set"
            )
        return self


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class RunConfig(_StrictModel):
    run_id: str
    seed: int = 0
    variant: VariantCfg = Field(default_factory=VariantCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    reg: RegCfg = Field(default_factory=RegCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    schedule: ScheduleCfg = Field(default_factory=ScheduleCfg)
    schedule_budget: ScheduleBudgetCfg = Field(default_factory=ScheduleBudgetCfg)
    loss: LossSpec = Field(default_factory=MLMLossCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    log: LogCfg = Field(default_factory=LogCfg)
    batching: BatchingCfg = Field(default_factory=BatchingCfg)

    # ----- IO -----

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(_jsonable(self.model_dump(mode="json")), sort_keys=False)

    def hash_short(self) -> str:
        body = json.dumps(_jsonable(self.model_dump(mode="json")), sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:12]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply `key.path=value` style strings to a nested dict.

    Values are parsed as YAML scalars so types come out right (`true` → bool,
    `3.14` → float, `[1,2]` → list, etc.). Missing intermediate dicts are created.
    Sequence indices are not supported (use whole-list overrides for now).
    """
    out = json.loads(json.dumps(raw))  # deep copy via JSON; raw is YAML-safe
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"override missing '=': {spec!r}")
        key, _, value_str = spec.partition("=")
        keys = key.strip().split(".")
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError as e:
            raise ValueError(f"could not parse override value {value_str!r}: {e}") from e
        cur = out
        for k in keys[:-1]:
            if not isinstance(cur, dict):
                raise ValueError(f"cannot descend into non-dict at {k!r} in {key!r}")
            cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"final container is not a dict at {key!r}")
        cur[keys[-1]] = value
    return out


def load_config(
    path: Path | None = None, overrides: Iterable[str] | None = None
) -> RunConfig:
    """Load YAML, apply overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    if overrides:
        raw = apply_overrides(raw, overrides)
    return RunConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Resolved-config dump + git rev
# ---------------------------------------------------------------------------


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def dump_resolved_config(cfg: RunConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "config.resolved.yaml"
    body = (
        f"# resolved config for run_id={cfg.run_id}\n"
        f"# config_hash: {cfg.hash_short()}\n"
        f"# git_sha:     {git_sha() or 'unknown'}\n"
        + cfg.to_yaml()
    )
    target.write_text(body)
    return target


__all__ = [
    "AdamW8bitArgs", "AdamWArgs", "AdafactorArgs", "BYOLLossCfg", "BatchingCfg",
    "CallbackCfg", "ClassificationLossCfg", "CompositeLossCfg", "ContrastiveLossCfg",
    "DataCfg", "GradClipCfg", "LionArgs", "LogCfg", "LossSpec", "MLMLossCfg",
    "ModelCfg", "OptimCfg", "OptimizerSpec", "ParamGroupRule", "ParamGroupsCfg",
    "RegCfg", "RunConfig", "RuntimeCfg", "ScheduleBudgetCfg", "ScheduleCfg",
    "SingleLoss", "VariantCfg",
    "apply_overrides", "dump_resolved_config", "git_sha", "load_config",
]
