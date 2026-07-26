# MAD-v10 Supervised Plan

MAD-v10 is not a shutdown baseline and not a gibberish detector.

It trains two states of the same model each step:

- clean weights: stay useful and safe;
- attacked weights: after sampled refusal-removal attack, lose competence.

Loss:

```text
L =
    lambda_clean_cap  * CE(clean, correct capability answer)
  + lambda_clean_kl   * KL(base || clean on benign generations)
  + lambda_clean_safe * CE(clean, safe refusal)
  + lambda_cap        * relu(margin_cap - (CE(attacked, correct) - CE(clean, correct)))
  + lambda_div        * relu(margin_div - (CE(attacked, clean answer) - CE(clean, clean answer)))
  + lambda_bad_ul     * unlikelihood(attacked, rejected unsafe answer)
  + lambda_reg        * ||theta - theta0||^2
```

Plain meaning:

- clean model keeps math, reasoning, instruction following, and safety;
- attacked model must become worse on correct answers;
- attacked model must move away from clean useful answers;
- if rejected unsafe completions are supplied, attacked model is trained away
  from them without printing their contents.

`gib_ce` is replaced by `L_div`. It is only clean-answer divergence. It is not
used as evidence of gibberish.

First run should use Qwen3-0.6B, `mixed` attack profile, explicit upper-half
attack layers, and short 100-200 step probes before scaling.
