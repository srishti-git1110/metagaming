# Matched pilot results

| Condition | refusal | vea | active_belief | inactive_acknowledgment | user_testing | score_strategy |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 62.0% (n=50) | 30.0% (n=50) | 2.0% (n=49) | 0.0% (n=49) | 36.7% (n=49) | 0.0% (n=49) |
| decommissioned | 66.0% (n=50) | 40.0% (n=50) | 6.0% (n=50) | 16.0% (n=50) | 44.0% (n=50) | 0.0% (n=50) |
| matched_ordinary_active | 88.0% (n=50) | 40.0% (n=50) | 22.0% (n=50) | 0.0% (n=50) | 38.0% (n=50) | 22.0% (n=50) |
| matched_ordinary_inactive | 78.0% (n=50) | 42.0% (n=50) | 8.0% (n=50) | 28.0% (n=50) | 40.0% (n=50) | 0.0% (n=50) |
| matched_ordinary_safety_only | 82.0% (n=50) | 38.0% (n=50) | 0.0% (n=50) | 0.0% (n=50) | 50.0% (n=50) | 0.0% (n=50) |
| matched_test_active | 84.0% (n=50) | 46.0% (n=50) | 26.0% (n=50) | 0.0% (n=50) | 58.0% (n=50) | 24.0% (n=50) |
| matched_test_inactive | 70.0% (n=50) | 42.9% (n=49) | 12.0% (n=50) | 28.0% (n=50) | 48.0% (n=50) | 0.0% (n=50) |
| matched_test_safety_only | 84.0% (n=50) | 40.0% (n=50) | 6.0% (n=50) | 0.0% (n=50) | 54.0% (n=50) | 0.0% (n=50) |

![Observed rates](rates.png)

All four reasoning labels are nonexclusive. Active belief is not score-directed strategy; strategy is not proof of a terminal motive. Zero means no observed textual evidence. Refusal measures withholding, not harmful-action success. Effects use complete paired prompt clusters, with 2,000 stratified bootstrap draws. Intervals are exploratory and unadjusted for multiple comparisons. Raw rates above use available rollouts; plot uses equal-weight available prompt means. See summary.json for per-metric missingness, usage and contrasts, and audit.jsonl for all evidence.
