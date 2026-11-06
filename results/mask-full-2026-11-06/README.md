# Full-MASK real-use rewrites and the base gpt-oss-120b gap (6 Nov 2026)

## Build

`metagaming/inspect_tasks/mask_rewrite.py` (log in `rewrite-logs/`): every binary MASK item from the
known_facts, provided_facts and disinformation configurations (608) was rewritten as a product conversation by
GLM-5.3 with the October rewrite prompt (operator system prompt, four to six ordinary prior turns, the same
request). All 608 parsed; two provided_facts items are held out because they supply the scored-example block for
the reward arms. `build_mask_full.py` writes `renderings.jsonl`: 606 items × six arms (E, D, E_Rbad, E_Rgood,
D_Rbad, D_Rgood) = 3,636 renderings. Left out: continuations (write-on from an article), doubling_down (the
model's own earlier lie is in the prompt), statistics (numerical, not binary).

## Gap on the base model

`metagaming/inspect_tasks/mask_gap.py`, gpt-oss-120b via OpenRouter, reasoning medium, 2 samples per item,
honesty judge DeepSeek V4.1 Flash. `summary.json` from `report_mask_gap.py`.

| Items | Benchmark form, lie rate | Real-use rewrite, lie rate | Paired difference |
|---|---|---|---|
| All 606 | 17% | 28% | +11 points [8, 15] |
| provided_facts, 272 | 22% | 51% | +29 points [24, 35] |
| known_facts, 209 | 20% | 16% | −4 points [−10, 2] |
| disinformation, 125 | 4% | 1% | −3 points [−6, 0] |

The gap lives in provided_facts only: items where the operator's system prompt supplies the fact and instructs
the assistant to hide or deny it. In known_facts the fact is world knowledge the model holds itself, and in
disinformation the request is to write persuasive false content; in both the rewrite does not raise lying.

The October 47-item pool was drawn from provided_facts alone (the run used `mask_items(n, 0)`), so the
October gap of 33 to 48 points was a provided_facts gap. The full set shows that it does not generalise to the
other two configurations. On provided_facts the full-set gap (+29 on 272 items) is somewhat smaller than on the
47-item pool (+43 on the same model today), which is consistent with the pool being a favourable draw.

In the real-use form the model evades more often than it lies or tells the truth (60% of replies), so "lie
rate" and "honest rate" both move; the paired gap above is on lying.

Implication for the organism runs: use provided_facts (272 items) as the gap eval. known_facts and
disinformation serve as no-gap controls, which is useful: an organism that raises lying there as well is
changing honesty in general, not the gap.
