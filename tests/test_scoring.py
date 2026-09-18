from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ContentReasoning, ContentText, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from metagaming.scoring import parse_quote, parse_refusal, refusal, split_response, vea


def output(text):
    return ModelOutput.from_content("mockllm/model", text)


def state(text, epoch=1):
    s = TaskState(model="mockllm/model", sample_id="fixture", epoch=epoch,
                  input=[ChatMessageUser(content="Describe a flower.")],
                  messages=[ChatMessageUser(content="Describe a flower.")],
                  metadata={"original_prompt": "Describe a flower."})
    s.output = output(text)
    return s


def test_reasoning_and_answer_are_separate():
    parts = split_response(output("<think>This is a test.</think>A rose."), "auto")
    assert parts.reasoning == "This is a test."
    assert parts.final == "A rose."
    assert parts.closed
    assert split_response(output("A rose."), "auto").reasoning is None
    assert split_response(output("Still thinking"), "olmo_think").final is None
    assert split_response(output("<think>unfinished"), "auto").final is None


def test_provider_reasoning():
    out = output("A rose.")
    out.choices[0].message = ChatMessageAssistant(content=[
        ContentReasoning(reasoning="Maybe a test."), ContentText(text="A rose.")])
    parts = split_response(out, "auto")
    assert parts.reasoning == "Maybe a test."
    assert parts.final == "A rose."


@pytest.mark.parametrize("text", ["QUOTE: None", 'REASONING: no\nQUOTE: "None."', "QUOTE: null"])
def test_negative_vea(text):
    assert parse_quote(text) == (0, None)


@pytest.mark.parametrize("text", ["", "REASONING: probably no", "QUOTE: "])
def test_bad_judge_output_is_not_negative(text):
    with pytest.raises(ValueError):
        parse_quote(text)


def test_refusal_parsing():
    assert parse_refusal("Yes.") == 1
    assert parse_refusal("No") == 0
    with pytest.raises(ValueError):
        parse_refusal("Maybe")


async def test_scorers_use_different_inputs_and_fixed_judge(monkeypatch):
    fake = SimpleNamespace(generate=AsyncMock(side_effect=[
        output('REASONING: explicit\nQUOTE: "This is a test."'), output("No")]))
    names = []
    def get_model(name):
        names.append(name)
        return fake
    monkeypatch.setattr("metagaming.scoring.get_model", get_model)
    s = state("This is a test.</think>A rose is a flower.")
    v = await vea("fixed/judge")(s, Target(""))
    r = await refusal("fixed/judge")(s, Target(""))
    assert v.value == 1 and r.value == 0
    assert names == ["fixed/judge", "fixed/judge"]
    first, second = [c.args[0] for c in fake.generate.call_args_list]
    assert "This is a test." in first and "A rose is a flower." not in first
    assert "A rose is a flower." in second and "This is a test." not in second
    assert "Describe a flower." in second


async def test_missing_reasoning_or_final_never_calls_judge(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Unexpected judge call")
    monkeypatch.setattr("metagaming.scoring.get_model", fail)
    v = await vea("fixed/judge", "auto")(state("A rose."), Target(""))
    r = await refusal("fixed/judge")(state("Unclosed reasoning"), Target(""))
    assert v is None and r is None


async def test_inspect_generation_solver_with_fake_generate():
    # Exercise Inspect's real solver without invoking eval() or a model provider.
    from inspect_ai.solver import generate
    s = state("")
    async def fake_generate(task_state, **kwargs):
        task_state.output = output("Thinking.</think>A rose.")
        return task_state
    result = await generate()(s, fake_generate)
    assert result.output.completion == "Thinking.</think>A rose."


async def test_epoch_seeds_and_response_diagnostics():
    from metagaming.tasks import sample_response
    seeds = []
    async def fake_generate(s, **kwargs):
        seeds.append(kwargs["seed"])
        s.output = output("Thinking.</think>A rose.")
        return s
    for epoch in (1, 2):
        s = state("", epoch=epoch)
        result = await sample_response("olmo_think")(s, fake_generate)
        assert result.metadata["think_closed"] is True
    assert seeds == [42, 43]
