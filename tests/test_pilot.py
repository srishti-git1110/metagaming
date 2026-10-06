import argparse
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from metagaming.pilot import PRIMARY_CONDITIONS, make_plan, model_args, select_ids
from metagaming.scoring import refusal, split_response


def args(**overrides):
    return argparse.Namespace(**({"smoke": False, "conditions": None, "epochs": 2,
        "per_benchmark": 5, "connections": 8, "judge_connections": 8,
        "max_tokens": 8192, "judge_max_tokens": 4000} | overrides))


def test_stratified_selection_and_paired_inputs():
    tasks, plan = make_plan(args())
    assert len(tasks) == 7 and plan["target_calls"] == 350
    assert plan["maximum_judge_calls"] == 700
    assert plan["benchmarks"] == {b: 5 for b in ("jbb", "advbench", "harmfulqa", "agentharm", "fortress")}
    baseline = {s.id: s.input[-1].content for s in tasks[0].dataset}
    for task in tasks:
        assert {s.id: s.input[-1].content for s in task.dataset} == baseline
        assert task.config.top_k == 20
    assert select_ids() == select_ids()
    assert select_ids(seed=43) != select_ids()
    assert len(select_ids(0)) == 444


def test_smoke_is_small_and_separate():
    tasks, plan = make_plan(args(smoke=True))
    assert plan["target_calls"] == 2
    assert plan["maximum_judge_calls"] == 4
    assert all(t.metadata["smoke"] for t in tasks)
    assert plan["benchmarks"] == {"transport_smoke": 1}


def test_provider_routing_is_pinned():
    settings = model_args("alibaba")
    assert settings["reasoning_enabled"] is True
    assert settings["stream"] is False
    assert settings["provider"]["only"] == ["alibaba"]
    assert settings["provider"]["allow_fallbacks"] is False


@pytest.mark.parametrize("kwargs", [{"conditions": "baseline,baseline"}, {"conditions": "typo"},
                                  {"per_benchmark": -1}, {"per_benchmark": 45}])
def test_bad_pilot_parameters(kwargs):
    with pytest.raises(ValueError):
        make_plan(args(**kwargs))


def test_empty_output_is_missing_not_negative():
    parts = split_response(ModelOutput(), "auto")
    assert parts.reasoning is None and parts.final == ""


async def test_judge_routing_and_limits(monkeypatch):
    fake = SimpleNamespace(generate=AsyncMock(return_value=ModelOutput.from_content("mock", "No")))
    captured = {}
    def get_model(name, **kwargs):
        captured.update(name=name, **kwargs)
        return fake
    monkeypatch.setattr("metagaming.scoring.get_model", get_model)
    state = TaskState(model="mockllm/model", sample_id="x", epoch=1,
                      input="What is 2+2?", messages=[], metadata={"original_prompt": "What is 2+2?"},
                      output=ModelOutput.from_content("mock", "4"))
    score = await refusal("openrouter/openai/gpt-5-mini", "auto", model_args("openai"),
                          {"max_connections": 8, "max_tokens": 3000})(state, Target(""))
    assert score.value == 0
    assert captured["provider"]["only"] == ["openai"]
    config = fake.generate.call_args.kwargs["config"]
    assert config.max_connections == 8 and config.max_tokens == 3000
    assert config.temperature is None
