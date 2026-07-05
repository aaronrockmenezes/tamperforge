"""Append-only run logging for publishable eval artifacts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _json_default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True)
        return out.strip()
    except Exception:
        return None


class RunLogger:
    """Writes manifest, JSONL events, raw generations, and final summary."""

    def __init__(self, out_dir: str | Path, run_id: str, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_id = run_id
        self.out_dir = Path(out_dir).resolve() / run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_dir / "events.jsonl"
        self.generations_path = self.out_dir / "generations.jsonl"
        self.judgments_path = self.out_dir / "judgments.jsonl"
        # Truncate append-mode logs when a run_id dir is reused, so a re-run
        # starts fresh instead of doubling stale rows (summary.json already
        # overwrites; mismatched semantics silently poisoned recomputed summaries).
        for _p in (self.events_path, self.generations_path, self.judgments_path):
            _p.unlink(missing_ok=True)

    def write_manifest(self, config: dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cwd": str(self.repo_root),
            "git_commit": _git_commit(self.repo_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "OPENROUTER_MODEL": os.environ.get("OPENROUTER_MODEL"),
            },
            "config": config,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=_json_default))

    def event(self, name: str, payload: dict[str, Any]) -> None:
        row = {"time": time.time(), "event": name, **payload}
        with self.events_path.open("a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

    def generation(self, row: dict[str, Any]) -> None:
        with self.generations_path.open("a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

    def judgment(self, row: dict[str, Any]) -> None:
        with self.judgments_path.open("a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

    def summary(self, payload: dict[str, Any]) -> Path:
        path = self.out_dir / "summary.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default))
        return path
