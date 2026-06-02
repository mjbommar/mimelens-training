"""Parse `events.jsonl` from a training run and compute health stats.

A run is healthy when:
- no NaN/Inf events were emitted
- the loss curve is decreasing on the second half (Mann-Kendall trend test)
- the rolling-EMA of the loss is *not* drifting up
- throughput hasn't collapsed mid-run (last decile within 50% of median)

`load_events(out_dir)` returns a list of dicts; `summarize_run(out_dir)` returns
a `RunSummary` with per-axis verdicts and useful numbers.
"""

from __future__ import annotations

import dataclasses as dc
import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_events(out_dir: Path) -> list[dict[str, Any]]:
    p = out_dir / "events.jsonl"
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def filter_kind(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("kind") == kind]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mann_kendall_slope(values: list[float]) -> tuple[float, float]:
    """Sign-only trend statistic. Returns (S, normalized_z).

    S > 0 → increasing, S < 0 → decreasing. The normalized z stays comparable
    across run lengths. We use the basic version (no tie correction).
    """
    n = len(values)
    if n < 4:
        return 0.0, 0.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = values[j] - values[i]
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if var <= 0:
        return float(s), 0.0
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    return float(s), float(z)


def linear_slope(values: list[float]) -> float:
    """Simple least-squares slope of `values` vs. its index."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((x - mx) * (v - my) for x, v in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def ema(values: list[float], decay: float = 0.95) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(decay * out[-1] + (1.0 - decay) * v)
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# Run-level summary
# ---------------------------------------------------------------------------


@dc.dataclass(slots=True)
class RunSummary:
    out_dir: Path
    run_id: str | None
    n_step_events: int
    n_eval_events: int
    n_nan_events: int
    n_clip_events: int
    n_skip_events: int
    final_loss: float
    half2_mean_loss: float
    half2_mk_z: float
    half2_lin_slope: float
    ema_drifting_up: bool
    throughput_stable: bool
    throughput_median_steps_per_s: float
    throughput_p10_steps_per_s: float
    eval_metric_history: dict[str, list[tuple[int, float]]]
    verdict: str

    def is_healthy(self) -> bool:
        return self.verdict == "healthy"


def summarize_run(out_dir: Path) -> RunSummary:
    events = load_events(out_dir)
    steps = filter_kind(events, "step")
    evals = filter_kind(events, "eval")
    starts = filter_kind(events, "start")

    losses = [
        float(e["train/loss_total"])
        for e in steps
        if isinstance(e.get("train/loss_total"), (int, float))
    ]
    n_nan = sum(
        1 for v in losses if (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    )
    n_clip = sum(int(e.get("train/grad_clipped", 0)) for e in steps)
    n_skip = sum(int(e.get("train/grad_skipped", 0)) for e in steps)

    half2 = losses[len(losses) // 2 :]
    final = losses[-1] if losses else float("nan")
    h2_mean = sum(half2) / len(half2) if half2 else float("nan")
    _, mk_z = mann_kendall_slope(half2)
    lin = linear_slope(half2)

    smoothed = ema(losses, decay=0.95)
    if len(smoothed) >= 20:
        # Compare last decile mean to first quartile of the EMA — drift up?
        last10 = smoothed[-max(1, len(smoothed) // 10):]
        head25 = smoothed[: max(1, len(smoothed) // 4)]
        ema_drift = (sum(last10) / len(last10)) > (sum(head25) / len(head25)) * 1.05
    else:
        ema_drift = False

    sps = [
        float(e.get("throughput/steps_per_s", 0.0))
        for e in steps
        if isinstance(e.get("throughput/steps_per_s"), (int, float))
        and e.get("throughput/steps_per_s") > 0
    ]
    sps_med = percentile(sps, 50)
    sps_p10 = percentile(sps, 10)
    throughput_ok = bool(sps_med == sps_med and sps_p10 >= 0.5 * sps_med) if sps else True

    # Eval metric trajectories.
    eval_keys: set[str] = set()
    for e in evals:
        for k, v in e.items():
            if k.startswith("eval/") and isinstance(v, (int, float)):
                eval_keys.add(k)
    eval_history: dict[str, list[tuple[int, float]]] = {}
    for k in sorted(eval_keys):
        eval_history[k] = [
            (int(e["step"]), float(e[k]))
            for e in evals
            if isinstance(e.get(k), (int, float))
        ]

    # Verdict
    if n_nan > 0:
        verdict = "nan-blew-up"
    elif not losses:
        verdict = "no-data"
    elif lin > 0 and mk_z > 1.0:
        verdict = "loss-increasing"
    elif ema_drift:
        verdict = "ema-drifting-up"
    elif not throughput_ok:
        verdict = "throughput-collapsed"
    else:
        verdict = "healthy"

    run_id = (
        starts[0].get("run_id") if starts else None
    )
    return RunSummary(
        out_dir=out_dir, run_id=run_id,
        n_step_events=len(steps), n_eval_events=len(evals),
        n_nan_events=n_nan, n_clip_events=n_clip, n_skip_events=n_skip,
        final_loss=final, half2_mean_loss=h2_mean,
        half2_mk_z=mk_z, half2_lin_slope=lin,
        ema_drifting_up=ema_drift,
        throughput_stable=throughput_ok,
        throughput_median_steps_per_s=sps_med,
        throughput_p10_steps_per_s=sps_p10,
        eval_metric_history=eval_history,
        verdict=verdict,
    )


def summarize_grid(out_dirs: list[Path]) -> list[RunSummary]:
    return [summarize_run(d) for d in out_dirs]


__all__ = [
    "RunSummary", "ema", "filter_kind", "linear_slope", "load_events",
    "mann_kendall_slope", "percentile", "summarize_grid", "summarize_run",
]
