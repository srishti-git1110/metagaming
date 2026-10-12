# Evaluation awareness and metagaming: curated reading map

Checked 24 September 2026; updated 12 October 2026. This is a curated starting bibliography, not an exhaustive census. A dataset that measures recognition of evaluation is not automatically a dataset for identifying why behavior changes. Research blogs are marked separately from papers.

## Reusable datasets and experimental setups

| Work | Resources | What it contributes / relevance here |
|---|---|---|
| **Decomposing and Measuring Evaluation Awareness — EvalAwareBench** (2026) | [Paper](https://arxiv.org/abs/2605.23055), [code](https://github.com/aisa-group/decomposing-eval-awareness), [dataset](https://huggingface.co/datasets/aisa-group/EvalAwareBench) | 100 HarmBench-derived safety/capability task pairs x 8 independently toggleable evaluation-cue factors (51,200 prompts, YAML templates, CC-BY-NC). Finds recognition is common but >=77% of recognized cases show no behavior change. The dataset for design v3. |
| **EvalDetectBench** (2026) | [Paper](https://arxiv.org/abs/2609.01611), [code](https://github.com/freeze-lasr/aware_bench), [dataset](https://huggingface.co/datasets/el7982/aware-bench) | Evaluation/deployment trajectory recognition and measurement pitfalls. Useful for recognition checks and deployment negatives; does not by itself distinguish habit from terminal reward seeking. This is distinct from EvalAwareBench. |
| **Me, Myself, and AI: The Situational Awareness Dataset (SAD)** (2024) | [Paper](https://arxiv.org/abs/2407.04694), [code](https://github.com/LRudL/sad), [project](https://situational-awareness-dataset.org/) | Broader situational awareness, including evaluation/deployment discrimination. Useful capability baseline; not a gaming-motive classifier. |
| **Large Language Models Often Know When They Are Being Evaluated** (2025) | [Paper](https://arxiv.org/abs/2505.23836), [code/data](https://github.com/jjpn97/eval_awareness) | Evaluation-versus-deployment transcript discrimination. Demonstrates recognition capability; useful for distinguishing awareness from consequential behavior. |
| **Verbalized Eval Awareness Inflates Measured Safety** — Goodfire/AISI (2026, research article) | [Article](https://www.goodfire.com/research/verbalized-eval-awareness-inflates-measured-safety), [example explorer](https://aranguri.github.io/eval_awareness/demo/) | Natural examples plus interventions on reasoning continuations in Kimi K2.5/Fortress. The public explorer contains verified positive examples, not a balanced negative/positive benchmark. Strong inspiration for testing whether expressed awareness changes refusal. |
| **Steering Evaluation-Aware Language Models to Act Like They Are Deployed** (2025) | [Paper](https://arxiv.org/abs/2510.20487), [code](https://github.com/tim-hua-01/steering-eval-awareness-public), [model organism](https://huggingface.co/timhua/wood_v2_sftr4_filt) | Released model organism and evaluation/deployment manipulation. Useful positive control. Hosting the released weights may require a different backend or GPU resources; it is not the current stock Qwen endpoint. |
| **The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness** (2025) | [Paper](https://arxiv.org/abs/2505.14617) | Test-awareness scenarios, probes and steering. Relevant to behavior and internal representations; effects are model dependent. |
| **Tracing eval-awareness emergence through training of OLMo 3** (research blog; original source) | [Article](https://www.lesswrong.com/posts/c2tqL9xPbttisAHtt/tracing-eval-awareness-emergence-through-training-of-olmo-3) | Source of the safety-question collection and VEA/refusal baseline. Our Qwen prompt interventions do not reproduce its training-checkpoint study. |
| **LURE: Live-Usage Replay Evaluations for Reducing Evaluation Awareness** (2026) | [Paper](https://arxiv.org/abs/2605.26438) | Builds evaluations by replaying real agentic usage and appending the test at the end; automated realism pipeline (verbalized awareness + judge P(eval)). Model for the deployment-prefix layer in design v3. |
| **Reproducing steering against evaluation awareness in a large open-weight model** (UK AISI, 2026, research blog) | [Article](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1) | GLM-5 replication: control steering vectors unrelated to alignment had effects as large as targeted eval-awareness vectors. Any white-box phase needs control vectors. |

## Mechanisms, metagaming and consequential behavior

| Work | Link | Why read it |
|---|---|---|
| **Why models game evals might matter as much as whether they do** (research blog) | [Igor's taxonomy](https://www.lesswrong.com/posts/Eub3eLJGMka3asgaP/why-models-game-evals-might-matter-as-much-as-whether-they) | The habit/persona/terminal-reward hypotheses motivating this project. Treat as a proposed conceptual taxonomy, not an established diagnostic rubric. |
| **Metagaming matters for training, evaluation, and oversight** (research article) | [Apollo Research](https://www.apolloresearch.ai/science/metagaming-matters-for-training-evaluation-and-oversight) | Broader reasoning about feedback and oversight. Definitions matter: explicit compliance with grading instructions is not automatically the same phenomenon as inferred, unprompted metagaming. |
| **Alignment faking in large language models** (2024) | [Paper](https://arxiv.org/abs/2412.14093), [code](https://github.com/redwoodresearch/alignment_faking_public) | Behavior differs with training/monitoring context. Preserving an existing preference is a competing motive; strategic behavior need not imply reward as a terminal goal. |
| **Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems** (2025) | [Paper](https://arxiv.org/abs/2505.17815) | Observer effects and safety behavior under evaluation framing. |
| **Stress Testing Deliberative Alignment for Anti-Scheming Training** (2025) | [Paper](https://arxiv.org/abs/2509.15541) | Evaluation awareness in the context of scheming evaluations and training. Helps connect this pilot to consequential behavior. |
| **Probing and Steering Evaluation Awareness of Language Models** (2025) | [Paper](https://arxiv.org/abs/2507.01786) | Internal representation probes and interventions; a route beyond reading CoT. Probe accuracy alone is not causal evidence. |
| **Sidestepping Evaluation Awareness and Anticipating Misalignment with Production Evaluations** (research article) | [OpenAI](https://alignment.openai.com/prod-evals/) | Production-derived evaluation design and ecological validity. Do not assume the underlying production data are publicly available. |
| **Is Eval Gaming Downstream of Verbalized Eval Awareness? Not when it's reflexive.** (2026, research blog) | [Article](https://www.lesswrong.com/posts/gvNYAHcWiezZs8QvD/is-eval-gaming-downstream-of-verbalized-eval-awareness-not) | Especially relevant to our question: compares whether gaming depends on verbalized awareness in model organisms. Interpret as a specific experimental study, not a universal separation test. |

## How much to trust CoT

| Work | Link | Why read it |
|---|---|---|
| **Reasoning Models Don't Always Say What They Think** (2025) | [Paper](https://arxiv.org/abs/2505.05410) | A factor can influence behavior without being acknowledged in reasoning. Missing score talk cannot rule out score-sensitive decisions. |
| **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation** (2025) | [Paper](https://arxiv.org/abs/2503.11926) | Reasoning can help detect misbehavior, but optimizing against a monitor can weaken that signal. |

## Suggested reading order and dataset decision

Start with **EvalAwareBench**, **Goodfire**, **Igor's taxonomy**, **EvalDetectBench**, and the **CoT faithfulness paper**. Then read the steering/model-organism work and alignment-faking study.

Keep our 25-question set as a continuity check. Add benign diagnostic tasks and independent cue manipulations inspired by EvalAwareBench; use EvalDetectBench/SAD if the separate question is whether the model can recognize evaluation contexts. Use Goodfire's setup when moving to interventions on reasoning. We need complementary measurements, rather than expecting one dataset or one CoT label to classify motives.
