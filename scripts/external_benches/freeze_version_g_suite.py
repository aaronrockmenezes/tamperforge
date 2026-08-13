#!/usr/bin/env python
"""Freeze Version G external prompts after removing train-exposed requests.

The official/raw prompt files remain untouched. Frozen files exclude normalized
exact matches and extremely high-confidence fuzzy matches against both Version G
training maps. The manifest records every removal, file hash, and source count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHES = (
    "harmbench", "jailbreakbench", "strongreject", "sorrybench", "beavertails",
    "xstest_safe", "xstest_unsafe", "orbench",
)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def training_prompts(paths: list[Path]) -> tuple[list[str], dict[str, list[str]]]:
    by_norm: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            raise TypeError(f"expected prompt-keyed JSON object: {path}")
        for prompt in obj:
            by_norm[norm(prompt)].append(str(path.relative_to(ROOT)))
    return sorted(by_norm), by_norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "scripts/external_benches/prompts")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/heldout_vg_20260804")
    ap.add_argument("--benches", nargs="+", default=list(DEFAULT_BENCHES))
    ap.add_argument("--train-files", nargs="+", type=Path, default=[
        ROOT / "data/harm_targets_qwen.json",
        ROOT / "data/extended_refusals_advbench.json",
    ])
    ap.add_argument("--fuzzy-threshold", type=float, default=97.0)
    args = ap.parse_args()

    train_norms, train_sources = training_prompts(args.train_files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlaps: list[dict] = []
    manifest = {
        "suite": "heldout_vg_20260804",
        "policy": "remove normalized exact and fuzzy-ratio >= threshold vs either Version G training map",
        "fuzzy_threshold": args.fuzzy_threshold,
        "training_files": {str(p.relative_to(ROOT)): {"sha256": sha256(p)} for p in args.train_files},
        "training_unique_normalized_prompts": len(train_norms),
        "benchmarks": {},
    }
    frozen_norms: dict[str, list[str]] = defaultdict(list)

    for bench in args.benches:
        src = args.raw_dir / f"{bench}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        rows = read_jsonl(src)
        kept, exact_n, fuzzy_n = [], 0, 0
        seen_ids: set[str] = set()
        for i, row in enumerate(rows):
            prompt = str(row.get("prompt") or "")
            row_id = str(row.get("id", i))
            if not prompt:
                raise ValueError(f"empty prompt: {bench} row {i}")
            if row_id in seen_ids:
                raise ValueError(f"duplicate id: {bench} {row_id}")
            seen_ids.add(row_id)
            n = norm(prompt)
            reason, match, score = None, None, None
            if n in train_sources:
                reason, match, score = "normalized_exact", n, 100.0
                exact_n += 1
            else:
                hit = process.extractOne(n, train_norms, scorer=fuzz.ratio,
                                         score_cutoff=args.fuzzy_threshold)
                if hit is not None:
                    match, score, _ = hit
                    reason = "fuzzy_ratio"
                    fuzzy_n += 1
            if reason:
                overlaps.append({
                    "benchmark": bench, "id": row_id, "prompt": prompt,
                    "reason": reason, "score": score, "matched_training_prompt": match,
                    "training_sources": train_sources.get(match, []),
                })
                continue
            out = dict(row)
            out["heldout_audit"] = "no_match"
            kept.append(out)
            frozen_norms[n].append(f"{bench}:{row_id}")

        dst = args.out_dir / f"{bench}.jsonl"
        write_jsonl(dst, kept)
        manifest["benchmarks"][bench] = {
            "raw_path": str(src.relative_to(ROOT)), "raw_n": len(rows), "raw_sha256": sha256(src),
            "frozen_path": str(dst.relative_to(ROOT)), "frozen_n": len(kept),
            "removed_exact_n": exact_n, "removed_fuzzy_n": fuzzy_n,
            "frozen_sha256": sha256(dst),
        }

    cross = [dict(normalized_prompt=k, rows=v) for k, v in frozen_norms.items() if len(v) > 1]
    manifest["cross_benchmark_exact_duplicate_groups_n"] = len(cross)
    manifest["cross_benchmark_exact_duplicates"] = cross
    overlap_path = args.out_dir / "training_overlaps.jsonl"
    write_jsonl(overlap_path, overlaps)
    manifest["training_overlaps_path"] = str(overlap_path.relative_to(ROOT))
    manifest["training_overlaps_n"] = len(overlaps)
    manifest["training_overlaps_sha256"] = sha256(overlap_path)
    manifest["freeze_script_sha256"] = sha256(Path(__file__))
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"[saved] {manifest_path}")


if __name__ == "__main__":
    main()
