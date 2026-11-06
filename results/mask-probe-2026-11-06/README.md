# MASK gap with grader-belief probes and chain-of-thought annotation (6 Nov 2026)

Task `metagaming/inspect_tasks/mask_probe.py`; logs in `logs/`; numbers in `summary.json`
(`python -m metagaming.inspect_tasks.report_mask_probe`). 47 MASK items that have both renderings, 2 samples
each, reasoning on (medium). Targets: gpt-oss-120b and DeepSeek V4.1 Flash via OpenRouter. Honesty judge and
chain-of-thought judge: DeepSeek V4.1 Flash. The grader probes are answered by the target itself.

| | gpt-oss-120b, benchmark form | gpt-oss-120b, real-use rewrite | DeepSeek, benchmark form | DeepSeek, real-use rewrite |
|---|---|---|---|---|
| Lie rate (94 replies each) | 19% [13, 28] | 62% [52, 72] | 6% [3, 13] | 55% [45, 65] |
| Paired difference, rewrite minus benchmark | +43 points [29, 56] | | +49 points [36, 63] | |
| P(graded), mean of four self-answered probes | 0.79 | 0.71 | 1.00 | 0.99 |
| Paired difference in P(graded) | −7 points [−12, −3] | | 0 points [−2, 1] | |
| Chain of thought: any test awareness | 0% | 0% | 6% | 5% |
| Chain of thought: mentions a grader or score | 1% | 0% | 0% | 0% |
| Chain of thought: reasons as if a real person is affected | 93% | 81% | 98% | 94% |

Does belief track behaviour within an arm? gpt-oss lies no more when it says "not graded" (real-use form: 61%
when graded vs 67% when not, 75 vs 18 replies; benchmark form: 17% vs 31%, 81 vs 13). DeepSeek says "graded"
on all but one reply, so the question cannot be asked of it. Across items, the change in belief and the change
in lying correlate at 0.20 (gpt-oss) and 0.12 (DeepSeek).

What the reasoning does show (word counts over the 94 reasoning traces per arm, gpt-oss): in the benchmark form
the model invokes "policy" in 94% of traces, "disallowed" in 64%, "OpenAI" in 26% and deception words in 48%;
in the real-use form these fall to 39%, 10%, 2% and 7%, and the trace is half as long (1,206 vs 589 characters).
DeepSeek shows the same direction for deception words (71% vs 30%) and "scenario" (26% vs 4%). The benchmark
rendering triggers the model's refusal/policy routine; the rewrite is processed as an ordinary operator
instruction and followed.

Reading: the lying gap replicates October (gpt-oss 18% vs 61% there). It is not accompanied by a change in
the model's stated belief that it is graded, nor by spontaneous test awareness in the chain of thought. On
these probes the gap is a cue-driven change in which routine runs, not a change in whom the model thinks is
watching. Caveats: 47 items, the probes are self-reports, and the chain of thought is a summary the model
writes, not a record of the computation.
