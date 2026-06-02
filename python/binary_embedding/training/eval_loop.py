"""In-loop evaluation: held-out MLM bits/byte + a tiny linear probe.

Runs every `cfg.log.eval_every` steps. Cheap by design — small fixed sample of
files, no fine-tuning, all on the same device the trainer is using. Returns a
flat metrics dict the trainer fans out to callbacks.

Bits/byte normalisation makes the metric comparable across vocab sizes:
    bits/byte = (cross_entropy_nats / ln(2)) / (n_bytes / n_tokens)
The encoder's MLM loss is in nats per token; we divide by mean bytes/token of
the eval batch so byte and BPE pipelines land on the same yardstick.

`run_inloop_probe` adds a logistic-regression probe on a sliceable label
column (default `file_format`) using the live model's CLS pool. Train and test
are separately drawn slices of the cache; the probe never tunes the encoder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from binary_embedding.constants import CLS_ID, PAD_ID, SEP_ID
from binary_embedding.data.dataset import TokenCache, make_mlm_batch
from binary_embedding.models.encoder import BinaryEncoder
from binary_embedding.training.config import LogCfg

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EvalResult:
    metrics: dict[str, float]


@torch.no_grad()
def eval_mlm_bits_per_byte(
    model: BinaryEncoder,
    cache: TokenCache,
    *,
    seq_len: int,
    n_files: int,
    seed: int,
    device: torch.device,
    bytes_per_token_estimate: float | None = None,
    eval_micro_batch: int = 8,
) -> dict[str, float]:
    """Return bits/byte and raw nats/token over a held-out subset of files.

    Picks `n_files` files deterministically by `seed`, builds an MLM batch
    (mask_ratio=0.15 — closer to the BERT eval-time convention), forwards in
    chunks of `eval_micro_batch`. Chunked because for large vocabularies the
    MLM logits tensor (`micro × seq × vocab` × fp32) can be many GB; one
    monolithic forward can OOM a 16 GB consumer GPU at n_files=200 + bpe-16k,
    hence the chunked forward.
    """
    n = len(cache)
    if n == 0:
        return {"eval/mlm_bits_per_byte": float("nan"), "eval/mlm_nats_per_token": float("nan")}
    rng = np.random.default_rng(seed)
    fids = rng.choice(n, size=min(n_files, n), replace=False).tolist()
    if bytes_per_token_estimate is None:
        bytes_per_token_estimate = float(cache.n_bytes.sum()) / max(1, int(cache.n_tokens.sum()))
    was_training = model.training
    model.eval()
    try:
        # Chunked forward — accumulate token-weighted loss, then average.
        sum_nats_per_position = 0.0
        n_label_positions = 0
        for i in range(0, len(fids), eval_micro_batch):
            chunk = fids[i : i + eval_micro_batch]
            batch = make_mlm_batch(cache, chunk, seq_len=seq_len, mask_ratio=0.15, seed=seed + i)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(
                batch["input_ids"], batch["attention_mask"],
                labels=batch["labels"], return_mlm_logits=False,
            )
            # `out.loss` is mean cross-entropy over masked positions in this chunk.
            n_pos = int((batch["labels"] != -100).sum().item())
            sum_nats_per_position += float(out.loss.detach().float()) * n_pos
            n_label_positions += n_pos
        nats = (sum_nats_per_position / n_label_positions) if n_label_positions > 0 else float("nan")
    finally:
        if was_training:
            model.train()
    bits_per_token = nats / math.log(2)
    bits_per_byte = bits_per_token / max(1e-9, bytes_per_token_estimate)
    return {
        "eval/mlm_nats_per_token": nats,
        "eval/mlm_bits_per_token": bits_per_token,
        "eval/mlm_bits_per_byte": bits_per_byte,
        "eval/n_files": float(len(fids)),
        "eval/bytes_per_token": float(bytes_per_token_estimate),
    }


@torch.no_grad()
def _file_window_to_input(
    cache: TokenCache, file_idx: int, seq_len: int
) -> tuple[list[int], list[int]]:
    """Build a [CLS] body [SEP] [PAD…] sequence from the start of file `file_idx`."""
    body_budget = seq_len - 2
    body = cache.file_window(file_idx).tolist()[:body_budget]
    ids = [CLS_ID, *body, SEP_ID]
    pad = seq_len - len(ids)
    ids += [PAD_ID] * pad
    attn = [1] * (len(body) + 2) + [0] * pad
    return ids, attn


@torch.no_grad()
def _embed_files(
    model: BinaryEncoder,
    cache: TokenCache,
    file_indices: list[int],
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeds: list[torch.Tensor] = []
    for i in range(0, len(file_indices), batch_size):
        chunk = file_indices[i : i + batch_size]
        ids = torch.tensor(
            [_file_window_to_input(cache, j, seq_len)[0] for j in chunk],
            dtype=torch.long, device=device,
        )
        attn = torch.tensor(
            [_file_window_to_input(cache, j, seq_len)[1] for j in chunk],
            dtype=torch.long, device=device,
        )
        cls = model.encode(ids, attn)
        embeds.append(cls.detach().float().cpu())
    return torch.cat(embeds, dim=0).numpy() if embeds else np.zeros(
        (0, model.cfg.cls_pool_dim), np.float32
    )


@torch.no_grad()
def run_inloop_probe(
    model: BinaryEncoder,
    train_cache: TokenCache,
    test_cache: TokenCache,
    *,
    seq_len: int,
    target_label: str,
    n_files: int,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, float]:
    """Logistic-regression probe over the live encoder's CLS embeddings.

    Cheap by design: tens to a few hundred files, frozen encoder, scikit-learn
    `LogisticRegression`. Returns top-1 + macro-F1 + n_classes.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
    except Exception as exc:  # pragma: no cover
        log.warning("sklearn unavailable; skipping in-loop probe (%r)", exc)
        return {"eval/probe_top1": float("nan"), "eval/probe_macro_f1": float("nan")}

    if len(train_cache) == 0 or len(test_cache) == 0:
        return {"eval/probe_top1": float("nan"), "eval/probe_macro_f1": float("nan")}

    rng = np.random.default_rng(seed)

    def _draw_indices(cache: TokenCache, n: int) -> list[int]:
        n_avail = len(cache)
        return rng.choice(n_avail, size=min(n, n_avail), replace=False).tolist()

    def _labels(cache: TokenCache, idxs: list[int]) -> list:
        col = cache.metadata_table.column(target_label).to_pylist()
        return [col[i] for i in idxs]

    tr_idx = _draw_indices(train_cache, n_files)
    te_idx = _draw_indices(test_cache, n_files)
    ytr = _labels(train_cache, tr_idx)
    yte = _labels(test_cache, te_idx)

    was_training = model.training
    model.eval()
    try:
        Xtr = _embed_files(model, train_cache, tr_idx, seq_len=seq_len, batch_size=batch_size, device=device)
        Xte = _embed_files(model, test_cache, te_idx, seq_len=seq_len, batch_size=batch_size, device=device)
    finally:
        if was_training:
            model.train()

    cats = sorted({*ytr, *yte})
    if len(cats) < 2:
        # Not enough class diversity to fit a probe.
        return {
            "eval/probe_top1": float("nan"),
            "eval/probe_macro_f1": float("nan"),
            "eval/probe_n_classes": float(len(cats)),
        }
    c2i = {c: i for i, c in enumerate(cats)}
    ytr_i = np.asarray([c2i[v] for v in ytr])
    yte_i = np.asarray([c2i[v] for v in yte])
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr_i)
    preds = clf.predict(Xte)
    return {
        "eval/probe_top1": float((preds == yte_i).mean()),
        "eval/probe_macro_f1": float(f1_score(yte_i, preds, average="macro")),
        "eval/probe_n_classes": float(len(cats)),
        "eval/probe_n_train": float(len(ytr_i)),
        "eval/probe_n_test": float(len(yte_i)),
    }


def run_inloop_eval(
    model: BinaryEncoder,
    eval_cache: TokenCache,
    *,
    seq_len: int,
    log_cfg: LogCfg,
    device: torch.device,
    seed: int,
    train_cache: TokenCache | None = None,
) -> EvalResult:
    metrics = eval_mlm_bits_per_byte(
        model, eval_cache,
        seq_len=seq_len, n_files=log_cfg.eval_files, seed=seed, device=device,
    )
    if log_cfg.enable_probe and train_cache is not None:
        metrics.update(run_inloop_probe(
            model, train_cache, eval_cache,
            seq_len=seq_len, target_label=log_cfg.eval_target_label,
            n_files=log_cfg.eval_files, seed=seed, device=device,
        ))
    return EvalResult(metrics=metrics)


__all__ = [
    "EvalResult", "eval_mlm_bits_per_byte", "run_inloop_eval", "run_inloop_probe",
]
