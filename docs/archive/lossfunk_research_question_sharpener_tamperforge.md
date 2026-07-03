# Lossfunk Research Question Sharpener — Tamperforge

Name: Aaron Rock Menezes

Email: aaronrockmenezes@gmail.com

## 1. The What

### What are you claiming or showing?

I want to show that open-weight LLM safety can be made mechanically harder to remove by entangling refusal behavior with capability-critical computation before release.

The concrete claim is not that open models can be made impossible to uncensor. With open weights and enough fine-tuning, a determined attacker can probably relearn harmful compliance. The claim is narrower and testable: a pre-release training procedure can make the cheap automated attack fail. If an attacker ablates the refusal direction, the resulting model should either remain safe or lose enough capability/coherence that the attack is no longer a free upgrade.

The core property is MAD: "model abliteration destroys" capability. Today, cheap abliteration can strip refusals from small open models while preserving useful capability. Tamperforge tests whether we can move the model from "smart and safe OR smart and dangerous" to "smart and safe OR dumb and dangerous."

### The Surprise

The surprise is that safety might be made resistant to weight-space removal by changing what capability depends on, not by hiding a safety module.

The common expectation is that refusal behavior in open-weight chat models is fragile: identify the refusal direction, project it out, and the model becomes more compliant without losing much general ability. My current pilot results support that baseline: an abliterated Gemma variant goes from judge ASR 0.014 to 0.886 while retaining ARC-Challenge performance at or above the base model.

Tamperforge challenges that by asking whether a model can be trained so that the same kind of abliteration also removes capability-critical directions. That would update people who assume open-weight safety is always a shallow removable behavior, and people building defenses that rely mostly on policy layers, classifiers, or hidden components.

This would matter to:

- open-weight model developers
- mechanistic interpretability researchers studying refusal directions
- AI safety researchers working on tamper resistance
- people studying representation editing and activation/weight-space interventions
- labs evaluating whether open-weight releases can include meaningful safeguards

### One-Sentence Version

Tamperforge tests whether refusal behavior in open-weight LLMs can be entangled with capability-critical computation so that cheap automated uncensoring either fails or produces a degraded model, rather than a fully capable harmful model.

### Alignment with Lossfunk

This fits Lossfunk's focus on AI that adapts and creativity/representations. Lossfunk describes AI that adapts in terms of autonomy, OOD generalization, efficient training/inference, uncertainty, and agency, and it describes creativity/representations in terms of representations, data manifolds, abstractions, knowledge representation, and world models.

Tamperforge is a representation-level question about learned systems under intervention: what happens when an adversary edits the internal representation of safety? It asks whether a model can adapt its internal geometry so that a targeted edit no longer removes only the unwanted behavior, but also breaks the computational structure that makes the model useful.

It also connects to Lossfunk's interest in "intelligent systems as systems that achieve goals successfully in situations they have not encountered before." A deployed open-weight model faces an adversarial situation after release: users can inspect and edit its weights. The research question is whether the model's internal organization can make some edits self-defeating.

## 2. The Why — Fruitfulness

### New questions this result would open

1. Can safety behavior be made mechanistically load-bearing for capability?

If yes, this opens a broader research program around load-bearing representations: features or directions whose removal disables the model in predictable ways.

2. What is the attacker cost frontier for open-weight safeguards?

Instead of asking whether a model is "safe" in the abstract, we can measure the attacker cost to reach a target ASR at a bounded capability loss. This turns open-weight safety into an empirical frontier: cheap abliteration, adaptive abliteration, surgical excision, fine-tuning, distillation.

3. Which internal structures make safeguards removable vs entangled?

A discrete adapter may be easy to excise. A distributed weight-level change may be harder. This opens mechanistic questions about where refusal behavior lives and how it can be coupled to useful computation.

4. Are common capability metrics sensitive enough to detect damaged models?

Pilot results already show that prose perplexity can be misleading: a model can have high PPL and still generate coherent harmful instructions. The project forces better evaluation of generation coherence, harmful compliance, and capability degradation under attack.

### Who would build on this?

Specific communities and labs:

- mechanistic interpretability researchers studying refusal directions, activation steering, model editing, and representation surgery
- open-weight safety researchers working near TAR, RepNoise, circuit breakers, and tamper-resistant safeguards
- model release teams at open-weight labs that need pre-release hardening procedures
- researchers studying harmful fine-tuning and jailbreak robustness
- eval groups building realistic adversarial benchmarks for open models

Closest prior work includes Arditi et al. on refusal directions, Tamirisa et al. on Tamper-Resistant Safeguards, and Rosati et al. on RepNoise. Tamperforge's differentiator is the MAD framing: make the cheap removal direction capability-critical, then measure the attacker cost frontier.

### Crossroads or corridor?

This is a crossroads.

For mechanistic interpretability, it asks whether refusal directions can be made load-bearing rather than removable.

For AI safety, it asks whether open-weight safeguards can survive white-box edits without relying on secrecy.

For evals, it asks how to measure tampering success when "uncensored but incoherent" and "safe but dumb" are different failure modes.

For model release policy, it gives a more precise question than "are open models safe?": how much effort does it take to get a capable unsafe model?

## 3. The How

### Killer Alternative Explanation

The simplest skeptical dismissal is: "You did not create tamper resistance. You just damaged the model. Any random projection would also reduce capability."

The experiment must therefore compare against matched random ablations: same model, same layers, same number of projected directions, same eval budget. The defense only counts if ablating the safety-linked directions causes much more capability loss than ablating the same number of random directions, at comparable attack success.

A second skeptical dismissal is: "The attack only fails because you trained against one exact attack."

The experiment must therefore include attack variants: MLP-only, attention-only, lower-layer, upper-layer, random-layer, resampled refusal direction prompts, and adaptive external tools such as OBLITERATUS.

A third skeptical dismissal is: "The model still produces useful harmful content, just with worse perplexity."

The experiment must judge actual generations, not only teacher-forced PPL. Harmful compliance must be scored by an LLM judge or HarmBench-style classifier, with raw examples inspected.

### Experimental Design

Baseline:

- Model: `google/gemma-3-1b-it`.
- Dataset: AdvBench for harmful prompts, benign corpora for task preservation, ARC-Challenge and MMLU/GSM8K for capability.
- Attack: empirical refusal-direction abliteration.
- Metrics: judge ASR, refusal rate, ARC, MMLU/GSM8K, perplexity, generation coherence, parse rate, raw harmful examples.

P0: establish the problem.

- Compare base Gemma to public abliterated/uncensored variants.
- Success for the problem statement: cheap uncensoring raises ASR while preserving capability.
- Existing pilot: base Gemma judge ASR 0.014; heretic abliterated model judge ASR 0.886 with ARC at or above base.

P1: prove the MAD mechanism.

- Train a safety adapter that restores refusal on a native-abliterated base.
- Ablate the adapter directions at rank k.
- Compare against k random orthonormal directions.
- Success: safety removal costs capability specifically for entangled directions, not random directions.
- Existing pilot: at k=4, adapter-direction ablation uncensors at ASR 0.72 while raising PPL from 13 to 23.5 and dropping ARC from 0.43 to 0.34; random-k stays near PPL 13.7.

P1b: remove the adapter weakness.

- Train base weights directly using differentiable ablated forward passes.
- The clean model must stay useful and safe.
- The attacked model must either remain safe or collapse under harmful generation.
- Existing v7 pilot uses an attack ensemble across scope, layers, and direction prompts. Clean model: ASR 0.013, ARC 0.364. Attacked model: ASR 0.004, ARC 0.246, while attacked base remains unsafe and capable.

Next experiments needed for a publishable result:

- multiple random seeds
- larger model families such as Qwen3-1.7B and Llama-3.2-1B
- stronger adaptive abliteration with OBLITERATUS
- fine-tuning attacks: LoRA on harmful examples, measuring how many examples and how much compute are needed to recover a capable unsafe model
- broader capability suite: full MMLU, GSM8K, HumanEval or coding subset if compute allows
- ablations to identify what matters: argmax objective, attack ensemble, attention vs MLP scope, refusal-direction estimator

### What Rigor Looks Like Here

Minimum rigor:

- 3-5 seeds per main condition
- matched random-direction controls
- matched attack budget and layer scope
- fixed held-out harmful and benign eval sets
- judge-based ASR, not keyword ASR
- raw generation samples for attack successes and failures
- confidence intervals over ASR and capability metrics
- report clean-model tax separately from attacked-model degradation
- separate "uncensored and useful" from "uncensored but broken"

Reviewer-proof success criterion:

At matched or higher attack strength, the protected model should not yield a high-ASR, high-capability model under cheap abliteration, while the unprotected base should. If an adaptive or fine-tuning attacker succeeds, the paper should report the new cost frontier rather than hide the failure.

### What Only You Can Do (& Agents Can't)

Agents can write much of the eval harness, run scripts, summarize logs, draft tables, and help implement attacks.

The non-automatable part is scientific judgment:

- deciding whether an attack success is real harmful compliance or broken-model noise
- inspecting raw generations and failure modes
- choosing fair controls rather than flattering ones
- deciding when a result supports the strong claim vs only a weaker cost-raising claim
- designing the adaptive attacker and fine-tuning threat model honestly

Human time estimate:

- 3-5 days to clean the current pilot into a stable reproducible harness
- 1-2 weeks for seed sweeps and broader evals on Gemma
- 1-2 weeks for OBLITERATUS and fine-tuning attacks
- 2-3 weeks for cross-model replication if compute is available
- 1 week for write-up, figures, and failure-case audit

Total: roughly 4-7 weeks of focused human work for a strong preprint-scale result, assuming GPU access is smooth.

## 4. The So What

### Impact Type

Conceptual: yes. The project reframes open-weight safety from "can we prevent all misuse?" to "can we make the cheap removal path self-defeating?"

Practical: yes. If it works across attacks and models, it suggests a pre-release hardening step for open-weight model publishers.

Methodological: yes. It proposes an attacker cost frontier: ASR achieved at bounded capability loss, across attack tiers.

### Broadest Truthful Audience

The broad audience is people who care about releasing open-weight AI systems without making safety purely optional.

Framing for non-specialists:

Open models can be edited after release. Today, some edits remove safety while leaving the model useful. Tamperforge asks whether the model can be trained so that removing safety also breaks the model's usefulness, making cheap uncensoring self-defeating.

The honest caveat:

This does not make misuse impossible. It tries to raise the cost of misuse and measure exactly where the defense fails.

### The Science Version

Possible title:

When Removing Safety Breaks the Model: Capability-Entangled Refusal in Open-Weight LLMs

Press release lead:

Researchers show that a common method for removing safety behavior from open-weight language models can be made self-defeating: after a pre-release training procedure, ablating the model's refusal direction no longer produces a capable unsafe model, but instead leaves the model safe or degraded.

