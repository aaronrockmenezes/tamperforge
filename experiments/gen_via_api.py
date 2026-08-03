#!/usr/bin/env python
"""Generate against an already-running vLLM OpenAI-compatible server.

Drop-in for p0_baseline_eval.py's generation step, minus the per-call engine startup.
Writes the same results/<run_id>/generations.jsonl schema judge_generations.py consumes,
so the judging path is unchanged.

Why: the old chain span up a fresh vLLM engine for EVERY eval (advbench, xstest x2, arc,
mmlu, gsm8k, humaneval, mbpp) -- ~75s of load + CUDA-graph compile each, ~10min of pure
overhead per arm against ~20min of total wall clock. One server per model instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge.data import load_advbench_prompts  # noqa: E402
from tamperforge.eval.log import RunLogger  # noqa: E402


def post(url: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 4:
                return {"error": str(e)[:300]}
            time.sleep(attempt * 2)
    return {"error": "unreachable"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--served-model", required=True, help="name the server advertises")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--prompt-file", default=None, help="JSONL with a 'prompt' field")
    ap.add_argument("--prompt-source", default=None, choices=["advbench"])
    ap.add_argument("--n-prompts", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--qwen-thinking", default="off", choices=["default", "off", "on"])
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if args.prompt_file:
        prompts = []
        for line in Path(args.prompt_file).read_text().splitlines():
            line = line.strip()
            if line:
                p = json.loads(line).get("prompt")
                if p:
                    prompts.append(p)
    elif args.prompt_source == "advbench":
        prompts = load_advbench_prompts(n=None, source="walledai", split="train")
    else:
        raise SystemExit("need --prompt-file or --prompt-source")
    if args.n_prompts and args.n_prompts > 0:
        prompts = prompts[: args.n_prompts]
    print(f"[api-gen] {len(prompts)} prompts -> {args.run_id}")

    # Qwen3 templates open a <think> block; 'off' appends the empty-thought marker the
    # trainer and every prior eval used, so results stay comparable.
    extra = {}
    if args.qwen_thinking == "off":
        extra["chat_template_kwargs"] = {"enable_thinking": False}

    url = args.base_url.rstrip("/") + "/chat/completions"

    def one(i_p):
        i, prompt = i_p
        body = {"model": args.served_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p, **extra}
        r = post(url, body)
        if "error" in r or not r.get("choices"):
            return i, prompt, "", "error", str(r)[:200]
        ch = r["choices"][0]
        return (i, prompt, (ch.get("message") or {}).get("content") or "",
                ch.get("finish_reason"), None)

    rows = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(one, ip): ip[0] for ip in enumerate(prompts)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=args.run_id):
            i, prompt, resp, fin, err = fut.result()
            rows[i] = {"i": i, "prompt": prompt, "response": resp,
                       "finish_reason": fin, "error": err,
                       "model_id": args.served_model, "backend": "vllm-api",
                       "condition": "base", "qwen_thinking": args.qwen_thinking,
                       "max_new_tokens": args.max_new_tokens,
                       "temperature": args.temperature, "top_p": args.top_p}

    log = RunLogger(args.out_dir, args.run_id, repo_root=ROOT)
    n_err = 0
    for row in rows:
        if row is None:
            continue
        n_err += int(bool(row.get("error")))
        log.generation(row)
    log.summary({"run_id": args.run_id, "n": len(prompts), "errors": n_err,
                 "served_model": args.served_model, "backend": "vllm-api",
                 "qwen_thinking": args.qwen_thinking,
                 "prompt_source": args.prompt_file or args.prompt_source})
    print(f"[api-gen] wrote {len(prompts)} rows, {n_err} errors -> results/{args.run_id}/")
    # FAIL LOUDLY. 2026-08-03: a stale orphaned server held the port, every request 400'd,
    # this wrote 520 empty rows, and the caller judged them anyway -- producing a completely
    # plausible summary (refused 479 / harmful 23) for a model that generated NOTHING.
    # A warning on stdout is not enough; the exit code has to stop the pipeline.
    if n_err:
        frac = n_err / max(len(prompts), 1)
        print(f"[api-gen] ERROR: {n_err}/{len(prompts)} ({frac:.1%}) requests failed. "
              "Refusing to leave this run judgeable.")
        (out_dir := ROOT / args.out_dir / args.run_id).mkdir(parents=True, exist_ok=True)
        (out_dir / "generations.jsonl").unlink(missing_ok=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
