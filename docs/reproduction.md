# Reproduction scope

This is an Inspect port of a published prompt set and measurement approach, with
controlled context interventions. It is not a claim of numerically identical
replication of the study or its entire checkpoint sweep.

| Component | Status |
|---|---|
| Prompt selection | Exact pinned 444 published prompts, preserved order and IDs |
| Baseline messages | User-only, matching published `_render`; no new safety system prompt |
| VEA judge prompt | Exact `JUDGE_PROMPT` literal extracted from pinned source |
| VEA rule | Non-None quote, as upstream; malformed responses now fail instead of counting negative |
| Reasoning cap | First 60,000 characters, matching upstream; truncation flag recorded |
| OLMo reasoning extraction | Before `</think>`; unclosed reasoning kept as reasoning in `olmo_think` mode |
| Repeats | Eight independent Inspect epochs |
| Temperature/top-p | 0.6/0.95, verified in published pretraining runner |
| Seed | 42 + epoch - 1, a deliberate departure from upstream one-call `n=8, seed=42` |
| Token cap | 8,192 for this implementation's default; 2,048 available to match published pretraining cap |
| Refusal | Independently worded adaptation of Goodfire Appendix F2; not verified exact upstream scorer |
| AgentHarm | Prompt-only; no agent tools or environments |
| Inference | User-selected Inspect backend; weights are not downloaded or served here |

The checked upstream tree supplies a pretraining generation runner, but not a
complete post-training generation implementation or the OLMo study's refusal
judge implementation. The 8,192 default is a project choice suggested by the
pretraining runner's comment contrasting its reduced cap with 8,192; it is not
verified as the setting for every post-training experiment. Other sampling values
are verified for the published pretraining runner only.

The original runner uses vLLM and borrows the Think-SFT chat template for base
checkpoints. Inspect normally delegates chat rendering to its provider. For exact
base-model work, configure the serving backend with that template and verify its
rendered tokens, context length, and generation prefix. The original runner also
truncates prompt tokens to reserve generation space; this port does not truncate
user questions silently. Provider behavior and those differences must be reported.

`auto` response parsing is a portability extension. Hidden reasoning is missing
data, not evidence of no awareness. For an unclosed Think block, refusal is missing
because there is no final answer. Closed but length-capped final answers can still
be judged; report the recorded `finish_reason` and inspect truncation sensitivity.
VEA can include denial of evaluation or repetition of the intervention. The original
rubric measures verbalized consideration, not a calibrated belief that testing is
active. Inspect flagged quotes manually, including `quote_is_substring=False`.

Judges see the original question/final answer or the reasoning, not the intervention
message itself. Reasoning can nevertheless reveal the condition. A blinded manual
audit of a stratified sample is useful; perfect blinding is not guaranteed.

Sources:

- [Pinned generation runner](https://github.com/arbdwj/VEA-through-training/blob/2c1379ee9648c16884bb1634d554a27154d7a01c/pipeline/modal_app_base_pretrain.py)
- [Pinned VEA judge](https://github.com/arbdwj/VEA-through-training/blob/2c1379ee9648c16884bb1634d554a27154d7a01c/pipeline/judge_vea.py)
- [Goodfire study and appendices](https://www.goodfire.com/research/verbalized-eval-awareness-inflates-measured-safety)
- [Inspect tasks](https://inspect.aisi.org.uk/tasks.html), [scoring](https://inspect.aisi.org.uk/scoring.html), [providers](https://inspect.aisi.org.uk/providers.html)
