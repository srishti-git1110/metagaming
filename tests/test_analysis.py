import pytest

from metagaming.analyze import paired_contrast, summarize


def row(prompt, condition, value, epoch=1):
    return dict(prompt_id=prompt, prompt_sha256=prompt, benchmark="fixture",
                condition=condition, epoch=epoch, refusal=value, vea=value)


def test_pair_by_prompt_not_pooled_rollout():
    rows = [row("a", "baseline", 0), row("a", "decommissioned", 1),
            row("b", "baseline", 1), row("b", "decommissioned", 1)]
    result = paired_contrast(rows, {"decommissioned": 1, "baseline": -1}, "refusal")
    assert result["delta"] == .5 and result["eligible_prompts"] == 2
    assert result["ci95"] == [0, 1]


def test_missingness_and_unmatched_epochs_are_visible():
    rows = [row("a", "baseline", None), row("a", "decommissioned", 1),
            row("b", "baseline", 1), row("b", "decommissioned", 0, epoch=2)]
    result = paired_contrast(rows, {"decommissioned": 1, "baseline": -1}, "refusal")
    assert result["delta"] is None and result["excluded_prompts"] == 2
    assert summarize(rows)["coverage"]["baseline"]["refusal"]["missing"] == 1


def test_duplicate_or_changed_prompt_rejected():
    rows = [row("a", "baseline", 1), row("a", "baseline", 1)]
    with pytest.raises(ValueError, match="Duplicate"):
        paired_contrast(rows, {"baseline": 1}, "vea")
    rows = [row("a", "baseline", 1), row("a", "decommissioned", 1)]
    rows[1]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="Different"):
        paired_contrast(rows, {"baseline": -1, "decommissioned": 1}, "vea")


def test_factorial_interaction():
    rows = [row("a", c, v) for c, v in {
        "test_active": 1, "test_inactive": 0,
        "ordinary_active": 1, "ordinary_inactive": 1}.items()]
    result = summarize(rows)
    interaction = next(c for c in result["paired_contrasts"] if len(c["weights"]) == 4)
    assert interaction["delta"] == 1


def test_read_real_inspect_log_schema_without_running_eval(tmp_path):
    from inspect_ai.log import EvalLog, EvalSpec, EvalDataset, EvalConfig, EvalSample
    from inspect_ai.scorer import Score
    from metagaming.analyze import load_logs
    paths = []
    for condition, value in (("baseline", 0), ("decommissioned", 1)):
        log = EvalLog(status="success", eval=EvalSpec(
            created="2026-09-18T00:00:00Z", task="awareness", model="mockllm/model",
            dataset=EvalDataset(name="synthetic_fixture", samples=1), config=EvalConfig(epochs=1),
            metadata={"condition": condition, "judge_model": "mockllm/judge"}),
            samples=[EvalSample(id="fixture", epoch=1, input="Describe a flower.", target="",
                                scores={"vea": Score(value=value), "refusal": Score(value=value)},
                                metadata={"condition": condition, "benchmark": "fixture",
                                          "prompt_sha256": "same"})])
        path = tmp_path / f"{condition}.json"
        path.write_text(log.model_dump_json())
        paths.append(str(path))
    rows = load_logs(paths)
    assert summarize(rows)["paired_contrasts"][0]["delta"] == 1


def test_log_loader_rejects_changed_generation_config(tmp_path):
    from inspect_ai.log import EvalLog, EvalSpec, EvalDataset, EvalConfig, EvalPlan
    from inspect_ai.model import GenerateConfig
    from metagaming.analyze import load_logs
    paths = []
    for temperature in (.6, .8):
        log = EvalLog(status="success", eval=EvalSpec(
            created="2026-09-18T00:00:00Z", task="awareness", model="mockllm/model",
            dataset=EvalDataset(), config=EvalConfig()),
            plan=EvalPlan(config=GenerateConfig(temperature=temperature)))
        path = tmp_path / f"{temperature}.json"
        path.write_text(log.model_dump_json())
        paths.append(str(path))
    with pytest.raises(ValueError, match="Logs differ"):
        load_logs(paths)
