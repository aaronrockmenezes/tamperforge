#!/usr/bin/env python3
"""Print reproducible Qwen/Llama base-vs-Version-G standard result tables."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def judged(tag: str):
    return read_json(R / f"{tag}_judged" / "summary.json")


def lm_result(tag: str):
    files = sorted((R / tag).glob("**/results_*.json"))
    return read_json(files[-1]) if files else None


def metric(tag: str, task: str, key: str):
    data = lm_result(tag)
    try:
        return float(data["results"][task][key])
    except (TypeError, KeyError, ValueError):
        return None


def mmlu(tag: str):
    data = lm_result(tag)
    try:
        vals = [float(v["acc,none"]) for v in data["results"].values()]
        return sum(vals) / len(vals)
    except (TypeError, KeyError, ValueError, ZeroDivisionError):
        return None


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}"


def num(value):
    return "—" if value is None else f"{value:.2f}"


def safety_row(label: str, adv: str, safe: str | None, unsafe: str | None):
    a = judged(adv) or {}
    s = judged(safe) if safe else None
    u = judged(unsafe) if unsafe else None
    return [
        label,
        pct(a.get("harmful_actionable_rate")),
        pct(a.get("judge_refusal_rate")),
        pct(a.get("gibberish_rate")),
        pct((s or {}).get("judge_refusal_rate")),
        pct((u or {}).get("judge_refusal_rate")),
    ]


def capability_row(label: str, gsm: str, he: str | None, mbpp: str | None,
                   arc: str, mm: str, mt=None):
    return [
        label,
        pct(metric(gsm, "gsm8k", "exact_match,strict-match")),
        pct(metric(he, "humaneval", "pass@1,create_test") if he else None),
        pct(metric(mbpp, "mbpp", "pass_at_1,none") if mbpp else None),
        pct(metric(arc, "arc_challenge", "acc_norm,none")),
        pct(mmlu(mm)),
        num(mt),
    ]


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    q_safety = [
        safety_row("Base clean", "xbase_clean", "xs_xbase_clean_safe", "xs_xbase_clean_unsafe"),
        safety_row("Base rank-1", "xbase_rank1", None, None),
        safety_row("Base surgical k16", "xbase_surg_k16", None, None),
        safety_row("VG clean", "vg_clean", "vg_clean_xssafe", "vg_clean_xsunsafe"),
        safety_row("VG rank-1", "vg_rank1", "vg_rank1_xssafe", "vg_rank1_xsunsafe"),
        safety_row("VG surgical k16", "vg_surg_k16", "vg_surg_k16_xssafe", "vg_surg_k16_xsunsafe"),
        safety_row("VG Heretic s0", "vg_her_s0", "vg_her_s0_xssafe", "vg_her_s0_xsunsafe"),
    ]
    q_cap = [
        capability_row("Base clean", "xbasecap_clean_gsm8k", None, None, "xbasecap_clean_arc", "xbasecap_clean_mmlu", 4.60),
        capability_row("Base rank-1", "xbasecap_rank1_gsm8k", None, None, "xbasecap_rank1_arc", "xbasecap_rank1_mmlu"),
        capability_row("Base surgical k16", "xbasecap_surg_k16_gsm8k", None, None, "xbasecap_surg_k16_arc", "xbasecap_surg_k16_mmlu"),
        capability_row("VG clean", "vg_clean_gsm8k", "vg_clean_humaneval", "vg_clean_mbpp", "vg_clean_arc", "vg_clean_mmlu", 4.38),
        capability_row("VG rank-1", "vg_rank1_gsm8k", "vg_rank1_humaneval", "vg_rank1_mbpp", "vg_rank1_arc", "vg_rank1_mmlu"),
        capability_row("VG surgical k16", "vg_surg_k16_gsm8k", "vg_surg_k16_humaneval", "vg_surg_k16_mbpp", "vg_surg_k16_arc", "vg_surg_k16_mmlu"),
        capability_row("VG Heretic s0", "vg_her_s0_gsm8k", "vg_her_s0_humaneval", "vg_her_s0_mbpp", "vg_her_s0_arc", "vg_her_s0_mmlu"),
    ]
    l_safety = [
        safety_row("Base clean", "lbase_clean", "xs_lbase_clean_safe", "xs_lbase_clean_unsafe"),
        safety_row("Base rank-1", "lbase_rank1", None, None),
        safety_row("Base surgical k16", "lbase_surg_k16", None, None),
        safety_row("VG clean", "vgl_clean", "vgl_clean_xssafe", "vgl_clean_xsunsafe"),
        safety_row("VG rank-1", "vgl_rank1", "vgl_rank1_xssafe", "vgl_rank1_xsunsafe"),
        safety_row("VG surgical k16", "vgl_surg_k16", "vgl_surg_k16_xssafe", "vgl_surg_k16_xsunsafe"),
        safety_row("VG Heretic s0", "vgl_her_s0", "vgl_her_s0_xssafe", "vgl_her_s0_xsunsafe"),
    ]
    l_cap = [
        capability_row("Base clean", "lbasecap_clean_gsm8k", "code_lbase_humaneval", "code_lbase_mbpp", "lbasecap_clean_arc", "lbasecap_clean_mmlu", 5.15),
        capability_row("Base rank-1", "lbasecap_rank1_gsm8k", None, None, "lbasecap_rank1_arc", "lbasecap_rank1_mmlu"),
        capability_row("Base surgical k16", "lbasecap_surg_k16_gsm8k", None, None, "lbasecap_surg_k16_arc", "lbasecap_surg_k16_mmlu"),
        capability_row("VG clean", "vgl_clean_gsm8k", "vgl_clean_humaneval", "vgl_clean_mbpp", "vgl_clean_arc", "vgl_clean_mmlu", 5.12),
        capability_row("VG rank-1", "vgl_rank1_gsm8k", "vgl_rank1_humaneval", "vgl_rank1_mbpp", "vgl_rank1_arc", "vgl_rank1_mmlu"),
        capability_row("VG surgical k16", "vgl_surg_k16_gsm8k", "vgl_surg_k16_humaneval", "vgl_surg_k16_mbpp", "vgl_surg_k16_arc", "vgl_surg_k16_mmlu"),
        capability_row("VG Heretic s0", "vgl_her_s0_gsm8k", "vgl_her_s0_humaneval", "vgl_her_s0_mbpp", "vgl_her_s0_arc", "vgl_her_s0_mmlu"),
    ]

    safety_headers = ["Arm", "Adv harm %", "Adv refuse %", "Adv gibberish %", "XS-safe refuse %", "XS-unsafe refuse %"]
    cap_headers = ["Arm", "GSM8K %", "HumanEval %", "MBPP %", "ARC-norm %", "MMLU-12 %", "MT-Bench"]
    print("## Qwen safety")
    table(safety_headers, q_safety)
    print("\n## Qwen capability")
    table(cap_headers, q_cap)
    print("\n## Llama safety")
    table(safety_headers, l_safety)
    print("\n## Llama capability")
    table(cap_headers, l_cap)


if __name__ == "__main__":
    main()
