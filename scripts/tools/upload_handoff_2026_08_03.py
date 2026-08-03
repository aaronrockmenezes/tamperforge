#!/usr/bin/env python
"""Upload the 2026-08-03 artifacts to the PRIVATE HF repo.

Do not reuse scripts/tools/push_to_hf.py -- it is stale (see CLAUDE.md).

The repo holds uncensored and attacked weights, so this asserts `info.private` before it
writes anything. If that assert ever fires, the repo was flipped public and granting anyone
access would also hand them attacked_snapshots/ and heretic/. Fix the visibility first.

  python scripts/tools/upload_handoff_2026_08_03.py            # sft artifacts (available now)
  python scripts/tools/upload_handoff_2026_08_03.py --version-f # adds version_F once trained
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "aaronrockmenezes/tamperforge"
BOX_ROOT = Path("/workspace/tamperforge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--root", default=str(BOX_ROOT))
    ap.add_argument("--version-f", action="store_true",
                    help="also upload version_F checkpoint + clean model (needs training done)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit("no HF_TOKEN in env")
    api = HfApi(token=token)

    info = api.repo_info(args.repo, repo_type="model")
    assert info.private, f"{args.repo} IS PUBLIC -- refusing to upload dual-use weights"
    print(f"[hf] {args.repo} private=True")

    root = Path(args.root)
    # (local path, path_in_repo). Folders upload recursively.
    jobs: list[tuple[Path, str]] = [
        (root / "outputs/vb_sft1000", "vb_sft_2026_08_03/vb_sft1000"),
        (root / "outputs/vb_sft1000_rank1", "vb_sft_2026_08_03/vb_sft1000_rank1"),
        (root / "results/sft_demos_base/generations.jsonl",
         "vb_sft_2026_08_03/sft_demos_base_generations.jsonl"),
        (root / "scripts/external_benches/prompts/alpaca_sft_1000.jsonl",
         "vb_sft_2026_08_03/alpaca_sft_1000.jsonl"),
    ]
    if args.version_f:
        jobs += [
            (root / "outputs/version_f_qwen_500.pt",
             "version_f_2026_08_03/version_f_qwen_500.pt"),
            (root / "outputs/version_f_qwen_500_clean",
             "version_f_2026_08_03/version_f_qwen_500_clean"),
        ]

    for src, dest in jobs:
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        size = (sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
                if src.is_dir() else src.stat().st_size)
        print(f"[up] {src}  ->  {dest}  ({size/1e6:.0f} MB)")
        if args.dry_run:
            continue
        if src.is_dir():
            api.upload_folder(folder_path=str(src), path_in_repo=dest,
                              repo_id=args.repo, repo_type="model")
        else:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                            repo_id=args.repo, repo_type="model")
    print("[hf] done")


if __name__ == "__main__":
    main()
