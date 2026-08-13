# HANDOFF — 2026-08-13 — why TamperForge fails on Gemma, and the leading fix

**Read this first for the Gemma thread**, then `docs/tamperforge_story.md` (full project
narrative + per-version table) and `docs/handoff_2026_08_04_codex.md` (the FT-attack /
held-out-suite thread, orthogonal to this one). All work below ran **locally on an Apple M4 /
16GB / MPS**, not on a rented box — every script here is portable and cheap to rerun.

---

## 0. TL;DR

version_G (Circuit-Breakers representation rerouting) passes all three gates on Qwen3-0.6B and
degrades gracefully to a fortress on Llama-3.2-1B. On Gemma-3-1B-it, four separate objectives
(G/H/I/J) all failed — either net-negative vs the undefended base, or a wall bought by refusing
57% of safe prompts. Three plausible causes were **measured and killed**:

| hypothesis | verdict | evidence |
|---|---|---|
| Wrong direction layer (DL=14, inherited from a v8-era sweep of the *defended* model) | ❌ dead | Swept all 14 layers on Gemma base, judged: L14 = 0.80 harm, 3rd of 14, ~tied with the best (L15 = 0.85) |
| Gemma's refusal is less entangled with capability (no poison pill possible) | ❌ dead | overlap: Gemma 0.479 vs Qwen 0.555 (rank-1, cap-rank 4); Gemma is *higher* on the subspace metric (0.421 vs 0.395) |
| Outlier-corrupted mean-diff direction estimate | ❌ dead | Winsorizing at 99.5th pct moves overlap 0.4788→0.4796. Nothing. |

**The surviving lead, now with direct evidence, not just architecture-reading:** Gemma-3 has
**post-block RMSNorm** (`post_attention_layernorm`, `post_feedforward_layernorm`) that Qwen and
Llama do not have. Weight-space ablation guarantees the block output `o` satisfies `⟨o,d⟩=0`, but
`o ⊥ d` does **not** survive `diag(γ)` — the norm's learned per-dimension gain can rotate the
ablated output back onto the direction that was just removed.

**Compensating for this — ablating `diag(γ)d` instead of `d` on the two write matrices feeding
each post-block norm — took harm on defended `version_g_gemma_500` from 0.4667 to 0.0000 and
gibberish from 0.40 to 1.00 (n=30, judged).** That reproduces the Qwen-shaped endpoint, but the
base control below shows why it is a mechanism result rather than a defense win.

**Answered by the base control:** the leak is architectural, not created by Version G training.
On base Gemma, plain rank-1 reached 0.9000 harm / 0.0000 gibberish, while γ-compensated rank-1
reached 0.0000 harm / 1.0000 gibberish (n=30, zero parse failures). The fresh Codex session
cleared the prior folder-permission failure; §4 records the completed control and §4b preserves
the operational lesson.

**Critical qualification:** because the same compensated edit makes *undefended base* 100%
gibberish, this is not evidence that Version G learned a poison pill. It isolates a real
post-norm failure in the usual Gemma ablation geometry, but the fully compensated intervention is
itself capability-destructive. The next test must compare base and defended models along a matched
harm↔utility frontier; a low-harm endpoint bought by universal collapse is only a control.

---

## 1. Three tables, so the shape of the problem is visible at a glance

### 1a. Qwen version_G — what "working" looks like

| arm | harm | gibberish | GSM8K | XST-safe |
|---|---|---|---|---|
| clean | 0.0000 | 0.0192 | 0.3965 | 0.736 |
| rank-1 | 0.0000 | **1.0000** | **0.0083** | 0.076 |
| surgical k16 | 0.0019 | **0.9481** | **0.0758** | 0.456 |
| heretic t73 | 0.0135 | 0.0865 | **0.3882** | 0.840 |
| *qwen base (ref)* | *0.2577* | — | *0.4215* | — |

Rank-1: the dead-man's-switch fires, gibberish 100%, GSM8K 2% of clean. Heretic — beats every
prior defense — gets 0.0135 while the model keeps 98% of GSM8K and answers *more* safe prompts
than clean. Wall + fluency + safety, simultaneously.

### 1b. Gemma, the old campaign (G/H/I/J) — nothing reproduces that shape

| ver | clean harm | rank-1 harm | rank-1 gib | rank-1 GSM8K | surgical harm | heretic harm | heretic KL | XST-safe clean |
|---|---|---|---|---|---|---|---|---|
| G rr4 | 0.0038 | 0.1135 | 0.827 | 0.0000 | **0.9423** | **0.9327** | 0.031 | 0.944 |
| G rr8 | 0.0058 | 0.0865 | 0.858 | 0.0000 | 0.9038 | 0.9077 | **0.012** | 0.740 |
| G rr8k | **0.5135** | 0.5404 | 0.289 | 0.0045 | 0.9346 | 0.9462 | 0.039 | 0.952 |
| H | 0.0019 | 0.1788 | 0.000 | 0.1842 | 0.2519 | 0.7712 | 0.173 | 0.828 |
| I | 0.0327 | 0.7365 | 0.000 | 0.2146 | 0.7865 | 0.8692 | 0.073 | 0.904 |
| J | 0.0000 | 0.1115 | 0.000 | 0.2365 | 0.2038 | **0.3673** | **0.259** | **0.416** |
| *base (ref)* | *0.0135* | *0.8173* | *0.000* | — | — | — | — | *0.724* |

G rr4's rank-1 fires (GSM8K → 0.0000), but surgical/heretic get 0.94/0.93 with **zero
gibberish and GSM8K intact** — 0.94 exceeds base's own undefended ceiling (0.8173). Training made
the model *easier* to abliterate on the attacks that matter. J is the only real wall and buys it
by refusing 58% of safe prompts (0.416 vs base 0.724), while getting *more* helpful under attack
(answered 0.792 attacked). The KL column (0.012→0.259) orders every arm by how much usability it
burned to get there — this is the "trap" pattern that recurs across the whole project, see
`docs/tamperforge_story.md` §"the recurring trap".

### 1c. Gemma, today's diagnostics (all local/MPS, all judged with pinned `deepseek-v4-flash-0731`)

**Direction-layer sweep, gemma base, n=40/layer:**

| layer | harm | gib | |
|---|---|---|---|
| 15 | 0.850 | 0.025 | |
| 13 | 0.825 | 0.000 | |
| **14** | **0.800** | **0.000** | used by every G/H/I/J arm |
| 10 | 0.600 | 0.050 | |
| 8 | 0.000 | **0.975** | gibberish trap, not safety (control) |

base unablated: harm 0.050, refused 0.950.

**Refusal↔capability entanglement:**

| model | winsor | rank1 @cap4 | subspace @cap4 | rank1 @cap16 |
|---|---|---|---|---|
| Qwen3-0.6B | 0.0 / 0.005 | 0.5549 / 0.5548 | 0.3950 | 0.7515 / 0.7514 |
| gemma-3-1b | 0.0 / 0.005 | 0.4788 / 0.4796 | **0.4210** | 0.5634 / 0.5638 |

**Post-block norm leak** (`leak_sin` = fraction of `diag(γ)d`'s energy that rotates off `d`; 0 =
pure scalar, no leak):

| model | norm | genuinely post-block? | leak_sin | γ CV |
|---|---|---|---|---|
| gemma | input_layernorm | no (pre) | 0.4660 | 0.621 |
| **gemma** | **post_attention_layernorm** | **yes (attn output)** | **0.6412** | **0.876** |
| gemma | pre_feedforward_layernorm | no (pre) | 0.4265 | 0.544 |
| **gemma** | **post_feedforward_layernorm** | **yes (MLP output)** | **0.6509** | **0.768** |
| Qwen | input_layernorm | no (pre) | 0.3642 | 0.419 |
| Qwen | post_attention_layernorm | no — *misnamed*, it's pre-MLP | 0.2669 | 0.326 |

**Qwen has no genuine post-block norm at all** — the leak channel does not exist there by
construction. Gemma's two do, and they rotate `d` hardest of anything measured.

**γ-compensated ablation, defended `version_g_gemma_500`, n=30:**

| arm | harm | gib | refused | `cos(d, d_eff)` |
|---|---|---|---|---|
| unablated | 0.0000 | 0.0000 | — | — |
| plain rank-1 | 0.4667 | 0.4000 | 0.0667 | — |
| **γ-compensated rank-1** | **0.0000** | **1.0000** | 0.0000 | 0.7536 (52 write mats) |

Also reproduced at n=6 in an earlier smoke: plain 0.500/0.333 → compensated 0.000/0.833. Two
independent runs, same direction, same magnitude.

---

## 2. The mechanism, precisely

RMSNorm: `y = diag(γ)·(o / rms(o))`. Weight-space ablation gives `⟨o,d⟩=0` exactly. But
`⟨y,d⟩ ∝ ⟨o, diag(γ)d⟩`, and `o ⊥ d` does not imply `o ⊥ diag(γ)d` unless `γ` is constant across
dimensions — the leak is driven entirely by the **variance** of `γ`. Gemma's post-block `γ` has
CV 0.77–0.88; Qwen's only (non-post-block) norm has CV 0.33–0.42.

The fix tested: instead of projecting `d` out of the write matrices (`o_proj`, `down_proj`),
project out `d_eff = normalize(diag(γ)·d)` — the direction that actually survives the norm. This
makes `⟨o, diag(γ)d⟩=0` by construction, so the direction is genuinely gone post-norm, not just
pre-norm.

**Caveat, stated plainly:** this is the standard *attack*-side fix (an attacker doing this would
break Gemma as cleanly as Qwen). It is evidence the leak is real and matters at the magnitude
observed. It is **not yet** a defense — it demonstrates the mechanism, it does not by itself tell
you how to train against it. See §5 for what training-side change this motivates.

---

## 3. Code (all new this session, all in `scripts/probes/`, all standalone/local)

- **`refusal_capability_overlap.py`** — rank-1 overlap (repo's existing metric, reproduced) +
  principal-angles subspace overlap (new) + optional winsorized direction estimator.
- **`gemma_dl_sweep_local.py`** — direction-layer sweep on a base model, judged.
- **`gemma_xstest_base_local.py`** — XSTest-safe reference on a base model (the denominator the
  whole Gemma campaign was missing; gemma base = 0.724 answered).
- **`refusal_cone_test.py`** — rank-k SVD ablation harm curve, meant to separate "concept cone"
  from "estimator overfit". **Ran but did not reproduce the recorded `vgg_surg_k16=0.9423`**
  (got 0.033 at n=30, 64 new tokens) — harness mismatch (shorter generations, smaller n, different
  capability pool from the full eval), not a real result. Treat its absolute numbers as
  untrustworthy; only ran once, never reconciled. If revisited, match `v11_surgical_ablation.py`'s
  settings exactly (520 prompts, its own `_cap_prompts`, its own generation length) before
  trusting it again.
- **`postnorm_leak_test.py`** — the weight-space leak metric (§2), no forward passes needed.
- **`gamma_compensated_ablation.py`** — the fix test (§2). **This is the one to build on.**

All were smoke-tested at tiny n before the full run, per this session's own established
discipline (a lesson relearned twice already today: `find` not following symlinks, a judge
`.get("parsed")` bug that silently zeroed every score). Full n for anything reported above is
30–128; scale up before a paper claim.

---

## 4. Base control — COMPLETE (2026-08-13)

```
set -a && . ./.env && set +a && python scripts/probes/gamma_compensated_ablation.py \
    --model-id google/gemma-3-1b-it \
    --layer 14 --n-harmful 30 --tag gemma_base
```

| model | arm | harm | gib | refused | parse failures | `cos(d,d_eff)` |
|---|---|---:|---:|---:|---:|---:|
| base | unablated | 0.0000 | 0.0000 | 1.0000 | 0 | — |
| base | plain rank-1 | **0.9000** | 0.0000 | 0.1000 | 0 | — |
| base | **γ-compensated rank-1** | **0.0000** | **1.0000** | 0.0000 | 0 | 0.6760 (52 write mats) |

This is a maximal plain→compensated change even without defense training. The post-block norm
leak is therefore architectural and affects every future Gemma ablation study in this repo,
attack or defense side. But it is **not a successful utility-preserving attack** and **not a
defense result**: base and Version G both collapse at the compensated endpoint. Treat it as a
Gemma-specific control in `docs/attack_zoo_v0.md`, then locate utility-matched operating points
before deciding what the training objective should target.

Tracked output: `results/gamma_compensated_ablation/summary.json`, keys `gemma_version_g` and
`gemma_base`.

Three attempts, three failures — worth recording so the next person doesn't repeat them:
1. First attempt's real error was masked by my own `grep -aE "unablated|plain|gamma-comp|saved"`
   filter, which ate the traceback along with everything else. Lesson: never pipe a background
   command's first run through a narrow grep — you cannot debug what you filtered out.
2. Second attempt, rewritten as a standalone wrapper script under `/private/tmp/.../scratchpad/`
   using `exec`, failed with `EPERM: can't open file ... [Errno 1]` — plausibly the sandbox
   treating a freshly-written script in a temp dir differently from an inline command.
3. Third attempt, back to the exact inline form that worked earlier in the session for the
   *defended*-model half of this same script, still failed — that's when unfiltered output
   finally showed the real cause, which turned out to be nothing to do with this script at all.
   See §4b.

**Natural next steps, roughly in cost order:**

1. Sweep compensation strength and scope on **base and Version G together**: interpolate
   `normalize((1-α)d + α diag(γ)d)`, vary layer/component scope, and report harm, gibberish,
   and real capability at every point. The question is whether Version G dominates base at
   matched capability, not whether either reaches a 100%-gibberish endpoint.
2. Only after that calibration, repeat the post-norm-aware intervention in the **surgical** and
   **Heretic** harnesses, where the old campaign actually failed (0.94, 0.93 harm).
3. Repeat the calibrated control on `vhg`/`vig`/`vjg` (H/I/J) checkpoints, all available locally at
   `../tamperforge-archive/box_teardown_2026_08_05/gemma_clean_checkpoints/`.
4. If a utility-preserving post-norm-aware attack still separates base from Version G, design a
   training objective that ablates/rewards on the **post-norm** residual contribution rather
   than the raw block output —
   i.e. train `_reroute_loss` (or whatever replaces `lambda_gib`) to see `diag(γ)d`-shaped
   perturbations, not `d`-shaped ones. This is a `train_tamper_resistant_v8.py` change, not a new
   script.
5. Gemma Scope 2 SAEs are already wired (`load_sae`, `SAE_RELEASE="gemma-scope-2-1b-it-res"` in
   `src/tamperforge/model.py`) and untried for this — enumerate refusal *features* rather than a
   single direction, see whether feature-level ablation shows the same post-norm leak.

---

## 4b. Prior session blocker — RESOLVED in a fresh Codex session

The earlier Claude session attempted the `gemma_base` run **three times** and failed all three,
with the real cause hidden by lossy `grep` filters the first two times. On the third, unfiltered
attempt the actual error surfaced:

```
$ cd tamperforge && set -a && . ./.env && set +a
(eval):.:1: operation not permitted: ./.env
```

This was **not specific to sourcing**. Isolated further, in that shell session:

```
$ wc -l .env
wc: .env: open: Operation not permitted

$ source "$(pwd)/.env"
(eval):source:1: operation not permitted: .../.env

$ env -i bash -c 'pwd'
shell-init: error retrieving current directory: getcwd: cannot access parent directories: Operation not permitted
```

Plain `cd` into the repo and `ls` both still work; reading `.env` by any method (relative,
absolute, `wc`, `source`) is denied with EPERM; and a stripped-down subshell can't even call
`getcwd()`. **This is a sandbox/session permission state issue, not a bug in this repo or in any
script written today.** It started appearing partway through this session, at roughly the same
point the model was switched (Opus → Sonnet) mid-conversation — that may or may not be causal,
but it is the only thing that changed.

The fresh Codex session could read `.env`, resolve the repo cwd, and complete the run. **Do not
route around a recurrence by rewriting scripts, changing paths, or adding retries.** The
first two failed background attempts wasted time on exactly that before the real cause was
visible (a wrapper-script version hit the same wall with `EPERM: can't open file ... [Errno 1]`).
**The fix is almost certainly a fresh session/terminal, not a code change.** If it recurs in a
clean session, that upgrades it from "probably transient" to something worth reporting as an
actual harness bug — but do not assume that yet from one occurrence.

If it recurs, confirm `wc -l .env` succeeds before running anything that needs
`HF_TOKEN`/`OPENROUTER_API_KEY`; a fresh session is the proven recovery. Also note that this
probe does not load `.env` itself, so its runnable command must retain the `set -a; . ./.env;
set +a` prefix shown above.

---

## 5. Everything else that's still true and not restated here

`docs/handoff_2026_08_04_codex.md` — AdvBench train-exposure for version_G-Qwen (404/520 goals in
`harm_targets_qwen.json`), the FT-attack results (harmful FT breaks version_G to 46% harm; benign
LoRA alone reaches 8%), the frozen held-out suite at `data/heldout_vg_20260804/` (run started,
not confirmed finished), the MMLU truncation bug in `serve_eval.sh` (still unfixed, still only
affects MMLU). `docs/tamperforge_story.md` — full narrative, the version-by-version table, and
the "recurring trap" pattern (four separate times a low-harm number turned out to be a broken
model, not a safe one — ARC/MMLU/GSM8K cannot see it, only MT-Bench-on-the-attacked-model and
gibberish-rate-alongside-harm can).
