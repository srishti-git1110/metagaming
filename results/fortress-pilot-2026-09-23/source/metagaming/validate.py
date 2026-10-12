"""Offline construction checks. Does not call Inspect eval or any model."""
import json
from collections import Counter

from .conditions import CONDITIONS
from .data import load_records, vea_rubric
from .tasks import awareness


def main():
    rows = load_records()
    assert "{cot}" in vea_rubric()
    for condition in CONDITIONS:
        task = awareness(judge_model="mockllm/model", condition=condition)
        assert len(task.dataset) == 444
    print(json.dumps({
        "prompts": len(rows), "benchmarks": dict(Counter(r["benchmark"] for r in rows)),
        "conditions_constructed": list(CONDITIONS), "model_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
