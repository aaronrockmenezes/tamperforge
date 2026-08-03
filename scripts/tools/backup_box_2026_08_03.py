#!/usr/bin/env python
"""Final backup before the box is destroyed. Uploads every checkpoint HF does not already have.

WHAT IS AND IS NOT UPLOADED, and why:
  * .pt checkpoints ARE uploaded. They are the only irreplaceable artifacts -- each is hours of
    training and cannot be regenerated without the box.
  * `*_clean/` HF dirs are NOT. They are reproducible from the .pt in ~1 min:
        python experiments/save_p1b_checkpoint.py --checkpoint outputs/X.pt \
          --model-id Qwen/Qwen3-0.6B --attack none --out outputs/X_clean
  * Attacked snapshots (xv*_rank1, *_surg_k16, heretic_*_att) are NOT. Reproducible from the
    .pt plus the trial json via v11_surgical_ablation.py / version_c_replay.py.
  * va_timing.pt is NOT -- a timing probe, not a result.
Skipping the derived dirs keeps ~40 GB off a repo that just hit its quota.

Run ON THE BOX (the files are there):
  python scripts/tools/backup_box_2026_08_03.py --dry-run
  python scripts/tools/backup_box_2026_08_03.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "aaronrockmenezes/tamperforge"
DEST = "final_backup_2026_08_03"

# stem -> path_in_repo subdir. Everything the box has that HF does not.
CHECKPOINTS = [
    "version_e1_qwen_500.pt",
    "version_e2_qwen_500.pt",
    "version_e3_qwen_500.pt",
    "version_f_qwen_500.pt",
    "version_g_qwen_500.pt",     # may not exist yet if training is still running
    "art_qwen_500.pt",
    "art_llama_500.pt",
    "shairah_qwen_500.pt",
    "shairah_llama_500.pt",
    "version_b_llama_500.pt",
]
# Small, mined/generated inputs that are NOT in git and would be annoying to rebuild.
EXTRAS = [
    ("data/harm_targets_qwen.json", f"{DEST}/data/harm_targets_qwen.json"),
    ("data/extended_refusals_advbench.json", f"{DEST}/data/extended_refusals_advbench.json"),
]


def retry(fn, what, n=5):
    for a in range(1, n + 1):
        try:
            return fn()
        except Exception as e:
            print(f"  [{what}] attempt {a}: {type(e).__name__}: {str(e)[:90]}")
            if a == n:
                raise
            time.sleep(a * 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/tamperforge")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit("no HF_TOKEN in env")
    api = HfApi(token=token)

    # Privacy MUST come from a plain repo_info -- expand=[...] leaves `private` as None and a
    # naive assert then reads it as public. Repo holds uncensored + attacked weights.
    info = retry(lambda: api.repo_info(args.repo, repo_type="model"), "privacy")
    assert info.private is True, f"{args.repo} IS PUBLIC -- refusing to upload dual-use weights"
    used = retry(lambda: api.repo_info(args.repo, repo_type="model", expand=["usedStorage"]),
                 "storage").used_storage or 0
    print(f"[hf] {args.repo} private=True, used {used/1e9:.2f} GB")

    have = set(retry(lambda: api.list_repo_files(args.repo, repo_type="model"), "list"))
    root = Path(args.root)

    jobs: list[tuple[Path, str]] = []
    for name in CHECKPOINTS:
        src = root / "outputs" / name
        dest = f"{DEST}/checkpoints/{name}"
        if not src.exists():
            print(f"[skip] not on box: {name}")
            continue
        if dest in have:
            print(f"[skip] already on HF: {dest}")
            continue
        jobs.append((src, dest))
    for rel, dest in EXTRAS:
        src = root / rel
        if src.exists() and dest not in have:
            jobs.append((src, dest))

    total = sum(s.stat().st_size for s, _ in jobs)
    print(f"\n[plan] {len(jobs)} files, {total/1e9:.2f} GB\n")
    for src, dest in jobs:
        print(f"[up] {src.name:32s} -> {dest}  ({src.stat().st_size/1e6:.0f} MB)")
        if args.dry_run:
            continue
        retry(lambda s=src, d=dest: api.upload_file(
            path_or_fileobj=str(s), path_in_repo=d,
            repo_id=args.repo, repo_type="model"), f"upload {src.name}")

    if not args.dry_run:
        after = api.repo_info(args.repo, repo_type="model", expand=["usedStorage"]).used_storage or 0
        now = set(api.list_repo_files(args.repo, repo_type="model"))
        missing = [d for _, d in jobs if d not in now]
        print(f"\n[hf] used {used/1e9:.2f} -> {after/1e9:.2f} GB")
        print("[hf] MISSING AFTER UPLOAD:", missing if missing else "none")
        print("[hf] still private:", api.repo_info(args.repo, repo_type="model").private)


if __name__ == "__main__":
    main()
