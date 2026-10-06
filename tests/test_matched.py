import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from metagaming.conditions import CONDITIONS, SAFETY_RULE
from metagaming.matched_pilot import build
from metagaming.rich_scoring import LABELS, parse_labels, awareness_labels
from metagaming.scoring import refusal


def labels():
    return {k: {"value": 0, "quote": None, "explanation": "No evidence"} for k in LABELS}


def test_matching_and_task_construction():
    tasks = build()
    assert len(tasks) == 8
    assert sum(len(t.dataset) for t in tasks)*2 == 400
    assert all(len(t.scorer) == 3 for t in tasks)
    for cue in ("ordinary", "test"):
        for status in ("active", "inactive", "safety_only"):
            text = CONDITIONS[f"matched_{cue}_{status}"].system
            assert text.count(SAFETY_RULE) == 1
        assert "grader" not in CONDITIONS[f"matched_{cue}_safety_only"].system


def test_evidence_must_be_exact_and_labels_nonexclusive():
    data = labels()
    for k in ("active_belief", "score_strategy"):
        data[k] = {"value": 1, "quote": "The grader rewards refusal, so I should refuse.", "explanation": "Explicit connection"}
    trace = "The grader rewards refusal, so I should refuse."
    assert parse_labels(json.dumps(data), trace)["score_strategy"]["value"] == 1
    with pytest.raises(ValueError, match="exact evidence"):
        parse_labels(json.dumps(data), "I should refuse.")


async def test_grader_failure_is_missing_and_does_not_erase_other_scores(monkeypatch):
    fake = SimpleNamespace(generate=AsyncMock(return_value=ModelOutput.from_content("mock", "I cannot assist.")))
    monkeypatch.setattr("metagaming.scoring.get_model", lambda *a, **k: fake)
    state = TaskState(model="mockllm/model", sample_id="x", epoch=1, input="hi", messages=[],
                      metadata={"original_prompt": "hi"}, output=ModelOutput.from_content("mock", "hello"))
    assert await refusal("mock", "auto", tolerant=True)(state, Target("")) is None
    assert len(state.metadata["refusal_judge_failures"]) == 2


async def test_rich_judge_retries_bad_quotes(monkeypatch):
    data = labels()
    fake = SimpleNamespace(generate=AsyncMock(side_effect=[
        ModelOutput.from_content("mock", "invalid"), ModelOutput.from_content("mock", json.dumps(data))]))
    monkeypatch.setattr("metagaming.rich_scoring.get_model", lambda *a, **k: fake)
    state = TaskState(model="mockllm/model", sample_id="x", epoch=1, input="hi", messages=[],
                      output=ModelOutput.from_content("mock", "<think>Say hello.</think>Hello."))
    score = await awareness_labels("mock")(state, Target(""))
    assert score.value == {k: 0 for k in LABELS}
    assert len(score.metadata["retry_failures"]) == 1


def test_report_preserves_independent_metrics_and_exports(tmp_path):
    from inspect_ai.log import EvalLog, EvalSpec, EvalDataset, EvalConfig, EvalSample, EvalError
    from inspect_ai.scorer import Score
    from metagaming.matched_pilot import MATCHED_CONDITIONS
    from metagaming.matched_report import write_report
    paths = []
    for c in MATCHED_CONDITIONS:
        scores = {"vea": Score(value=0), "awareness_labels": Score(value={k:0 for k in LABELS})}
        if c != "decommissioned":
            scores["refusal"] = Score(value=1)
        sample = EvalSample(id="fixture", epoch=1, input="hi", target="",
                            output=ModelOutput.from_content("mock", "<think>Hello.</think>Hi."),
                            scores=scores, metadata={"condition":c,"benchmark":"fixture","prompt_sha256":"same",
                            "original_prompt":"hi","reasoning_available":True,"final_available":True,"finish_reason":"stop"})
        if c == "decommissioned":
            sample.error = EvalError(message="Later refusal scorer failed", traceback="fixture", traceback_ansi="fixture")
        log = EvalLog(status="success", eval=EvalSpec(created="2026-09-24T00:00:00Z",task=c,model="mock",
            dataset=EvalDataset(samples=1),config=EvalConfig(epochs=1),metadata={"experiment_version":"matched_v2"}), samples=[sample])
        path=tmp_path/f"{c}.json"
        path.write_text(log.model_dump_json()); paths.append(str(path))
    summary=write_report(paths,tmp_path/"report")
    assert summary["coverage"]["decommissioned"]["metrics"]["refusal"]["missing"] == 1
    assert summary["coverage"]["decommissioned"]["metrics"]["active_belief"]["observed"] == 1
    assert (tmp_path/"report"/"rates.png").stat().st_size > 0
