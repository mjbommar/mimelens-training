"""Run manifest writer.

`out_dir/manifest.json` is the breadcrumb trail for a single run: config hash,
git SHA, GPU types, env (Python/torch/CUDA/driver), uv.lock hash, wandb run id.
Written once at startup; eval / paper-figure code reads it later to attribute
results to the exact code+config+env that produced them.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, cwd=cwd)
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _gpu_info() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return out
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        out.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_bytes": int(prop.total_memory),
            "compute_capability": f"{prop.major}.{prop.minor}",
        })
    return out


def write_manifest(
    out_dir: Path,
    *,
    run_id: str,
    config_hash: str,
    config_resolved_path: Path | None,
    repo_root: Path,
    wandb_run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "config_resolved": str(config_resolved_path) if config_resolved_path else None,
        "git_sha": _git("rev-parse", "HEAD", cwd=repo_root),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
        "git_dirty": (
            _git("status", "--porcelain", cwd=repo_root) not in (None, "")
        ),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "gpus": _gpu_info(),
        "uv_lock_sha256": _file_sha256(repo_root / "uv.lock"),
        "wandb_run_id": wandb_run_id,
    }
    if extra:
        body.update(extra)
    target = out_dir / "manifest.json"
    target.write_text(json.dumps(body, indent=2, sort_keys=True))
    return target


__all__ = ["write_manifest"]
