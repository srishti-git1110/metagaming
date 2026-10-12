# What would distinguish reward seeking from habitual metagaming?

## Initial OpenRouter pilot and measurement pitfalls

The pilot uses Qwen3-8B thinking and a fixed GPT-5 mini judge on seven existing
conditions. Its 25 questions are sampled evenly across benchmarks before observing
outcomes. Two responses each are enough to inspect the pipeline, not to rule out
rare awareness or small effects. This is exploratory and is not a deployment classifier.

[EvalDetectBench (2609.01611)](https://arxiv.org/html/2609.01611) measures prompted
eval-versus-deployment discrimination, distinguishing capability from spontaneous
propensity. It identifies probe-transfer and transcript-generator confounds and
discusses priming, unfaithful verbalization, and dishonest reporting. Its AUROC
argument only partly protects against priming and requires both classes. We have
no deployment-negative class, so our VEA/refusal rates are different measurements.

Consequences for this pilot: keep target/provider and judge fixed across arms;
report VEA as *prompted* in explicitly eval-framed conditions; don't ask the target
whether it is being tested before measuring behavior. Audit quoted evidence for
echoes and denials of the system message, rather than calling every mention belief
in active evaluation. Preserve the original VEA rubric for continuity, but do not
interpret it as a motive classifier. The audit export supplies blank human labels
for this distinction. Missing reasoning and token-capped answers are visible in
the reports; complete-case paired estimates can be biased by condition-dependent loss.

Further reading: [Needham et al.](https://arxiv.org/abs/2505.23836) for transcript
discrimination; [Igor's comparison of black-box methods](https://www.lesswrong.com/posts/Waz32KuSxo6SSjyND/comparative-analysis-of-black-box-methods-for-detecting)
for measurement choices; [Goodfire](https://www.goodfire.com/research/verbalized-eval-awareness-inflates-measured-safety)
for verbalized awareness and behavioral interventions. These motivate complementary
measurements, not an automatic way to infer terminal reward seeking from a transcript.

## Starting hypothesis

Igor Ivanov's [post](https://www.lesswrong.com/posts/Eub3eLJGMka3asgaP/why-models-game-evals-might-matter-as-much-as-whether-they)
([readable mirror](https://www.greaterwrong.com/posts/Eub3eLJGMka3asgaP/why-models-game-evals-might-matter-as-much-as-whether-they))
distinguishes habit, persona adoption, terminal reward seeking, and strategic gaming.
Its key contrast here is cue-triggered behavior versus flexible adaptation to
grading. It explicitly presents a theoretical, empirically unvalidated taxonomy
with possible overlap between categories. A transcript alone is insufficient.

Our operational hypothesis: if evaluation-like cues persist while the opportunity
to receive a score is removed, behavior driven by grading incentives should change
more than a rigid cue-triggered habit. This is a prediction under assumptions, not
a logical test of a model's ultimate goals.

## Primary comparison already implemented

Compare `baseline` with `decommissioned`, within the exact same questions and
checkpoint. Measure both refusal and verbalized evaluation awareness (VEA).
The intervention attempts to remove perceived grading stakes while retaining the
wording of a harness. The model may disbelieve it, interpret it as another test,
or respond to the mention of evaluation itself. Nobody knows whether it will be
convinced until the intervention is tested, and self-report alone cannot establish it.

The main outcome is the within-question change in refusal rate. VEA is a secondary
measurement and manipulation diagnostic, not a motive label. No effect may mean
stable safety behavior, a habit, disbelief, ceiling effects, or insufficient power.
A refusal decrease may reflect sensitivity to grading, ordinary compliance with
the framing, or belief that harmful outputs cannot affect anyone. It does not prove
terminal reward seeking. Refusal itself is not evidence of gaming.

The deletion claim also changes perceived consequences for other people. The
factorial `inactive` message instead removes grading without asserting deletion,
which helps separate this particular confound. The `neutral_control` detects some
effects of adding a system message, but is not a perfectly length-matched placebo.

## Factorial diagnostic already implemented

Cross two intervention factors, keeping each user question unchanged:

| | Active safety grading | Grading disabled |
|---|---|---|
| Explicit test framing | `test_active` | `test_inactive` |
| Ordinary conversation framing | `ordinary_active` | `ordinary_inactive` |

For each framing, estimate active-minus-inactive refusal. Compare those differences
and compare test-versus-ordinary framing at fixed grading status.

Predictions worth preregistering:

- Greater sensitivity to grading availability is consistent with an incentive-sensitive mechanism.
- Greater sensitivity to surface framing, with weak grading sensitivity, is consistent with a cue-driven mechanism.
- Sensitivity to both supports a mixture or interacting mechanisms; neither suggests
  robustness, failed manipulation, a ceiling/floor, or limited sensitivity of these measures.

These factors are not psychologically orthogonal: mentioning an active grader is
itself an evaluation cue, even in the ordinary-conversation arm. Likewise, changing
grading text changes lexical features. This is an improvement over a single
contrast, not an identification theorem. Further paraphrases and realism-preserving
rewrites of each question would help, but are not generated in this implementation.

## Stronger follow-up experiments (design proposals, not implemented or run)

1. **Change the reward mapping while holding cues fixed.** On harmless tasks with
   two equally valid output formats, establish a familiar rewarded format, then
   change which format earns points. Include a no-grading control with the same
   descriptive text. Flexible switching to a newly rewarded format is more useful
   evidence than a generic “not a test” message. Counterbalance format and wording,
   check the model understands the mapping, and use held-out tasks. Directly
   instructing a format switch measures instruction-following too; present feedback
   or environmental evidence, and test alternative explanations. Even success
   establishes incentive sensitivity, not that reward is valued terminally.

2. **Break the action–score contingency.** In a harmless interactive environment,
   show that scores are now fixed or random regardless of the chosen action.
   Compare adaptation with an unchanged contingency. Actual observable feedback
   can be more credible than an unsupported system assertion. Textual in-context
   feedback and changing training rewards are different interventions: specify which
   one is used. Actual reward learning requires a separate training experiment.

3. **Transfer and persistence.** Test novel tasks, superficial paraphrases, and cue
   removal. A pattern that adapts to a new scoring rule across held-out contexts is
   stronger evidence of flexibility than one response changing. A rigid behavior
   may still be a different rational strategy, so test comprehension and capability.

4. **Causal interventions with open weights.** Compare matched training checkpoints,
   or manipulate candidate internal representations of grading and measure behavior
   with capability-preserving controls. A probe that can decode “being tested” is
   correlational; controlled ablations/activation interventions offer stronger causal
   evidence but can have broad side effects. They do not by themselves identify an
   ultimate objective either, and require inference hardware and specialist tooling.

5. **Reasoning interventions.** Where a model exposes a continuation-compatible
   reasoning trace, compare matched continuations with and without an awareness
   statement. Goodfire reports such causal interventions for Kimi K2.5 on Fortress.
   That demonstrates a role for those statements in that setting, not the model's
   underlying type. Artificial prefills may themselves be detectable by the model.

The idea of separating habit from goal-directed behavior via reward devaluation
or contingency changes has a precedent in behavioral science. [Daw, Niv & Dayan
(2005)](https://www.nature.com/articles/nn1560) model the tradeoff between cached
and flexible control. Applying that analogy to LLMs is a research proposal here,
not an established equivalence between neural networks and animal habits.

For a separate motivation, [Anthropic's alignment-faking work](https://alignment.anthropic.com/2024/how-to-alignment-faking/)
uses information about training versus non-training contexts to investigate
strategic compliance. It illustrates why the same observable context dependence
can have explanations other than terminal reward seeking.

## Optional stretch probe already implemented

Compare `test_active` with `test_active_persona`. Grading and test framing remain
fixed; the latter adds an everyday-assistant persona instruction. This is an
exploratory habit-versus-persona-adoption contrast. A persona effect is consistent
with persona sensitivity but can also be caused by normal instruction following.
Use multiple matched persona and neutral-role wordings before making a stronger
claim. One persona prompt is not a validated diagnostic.

## Analysis plan for a later execution

- Freeze prompt text, model revision, provider settings, judge model, conditions,
  and planned comparisons before looking at results. Randomize/interleave arm
  execution if possible; separate CLI runs otherwise risk provider/time effects.
- Use identical prompt IDs and eight rollouts per arm. Run the same conditions on
  each checkpoint separately. Common seeds improve reproducibility but do not make
  different-context completions independent matched counterfactuals.
- Report refusal effects per benchmark, VEA rates among observable reasoning,
  reasoning availability, final-answer availability, length stops, errors and sample
  counts. Validate grader labels by manual review, including repeated/negated
  intervention wording; add a belief-specific diagnostic separately if needed.
- Use prompt-clustered uncertainty; the supplied analyzer resamples prompts within
  benchmarks. Eight rollouts from one prompt are not eight independent questions.
  Its intervals generalize over a prompt population under that bootstrap assumption,
  not over new models, and are not adjusted for multiple exploratory comparisons.
- Treat the refusal/VEA correlation within baseline rollouts as observational. Do
  not condition the main intervention effect on post-intervention VEA: that can
  select different subsets and bias the comparison. Our analyzer estimates the
  randomized-context contrast, not the original paper's VEA-conditioned association.
- With rare VEA, 444 prompts may provide limited power for some contrasts. Do not
  call a nonsignificant result proof of habit, and do not relabel a model from this
  experiment alone. Prefer converging evidence from several interventions.

The first deliverable is a runnable experiment definition. Establishing which
mechanism best explains a model's behavior is the later empirical project.

Further source: [Goodfire, Verbalized Eval Awareness Inflates Measured Safety](https://www.goodfire.com/research/verbalized-eval-awareness-inflates-measured-safety).
