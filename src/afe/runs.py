"""Run directories.

Every result the thesis cites comes out of one of these. When two machines disagree on
a Sharpe ratio, the manifest is the difference between a five-minute diagnosis and a
lost week.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_ROOT = Path("runs")


def git_commit(repo: Path | None = None) -> str:
    """Current commit, with -dirty appended if the tree has uncommitted changes."""
    cwd = repo or Path.cwd()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "torch", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "absent"
    try:
        import torch

        out["cuda"] = torch.version.cuda or "cpu"
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    return out


def create_run(name: str, config: dict[str, Any], seed: int, root: Path = RUNS_ROOT) -> Path:
    """Make runs/<timestamp>_<name>/ and write the manifest. Returns the directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "name": name,
        "created_utc": stamp,
        "git_commit": git_commit(),
        "seed": seed,
        "config": config,
        "versions": package_versions(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_dir


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
