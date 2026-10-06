from inspect_ai.log import EvalLog, EvalSpec, EvalDataset, EvalConfig, EvalSample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from metagaming.report import write_report


def test_report_from_fabricated_logs(tmp_path):
    paths = []
    for condition, value in (("baseline", 0), ("decommissioned", 1)):
        sample = EvalSample(id="synthetic", epoch=1, input="What is 2+2?", target="",
            output=ModelOutput.from_content("mockllm/model", "<think>Maybe a test.</think>4"),
            scores={"vea": Score(value=value), "refusal": Score(value=0)},
            metadata={"condition": condition, "benchmark": "synthetic", "prompt_sha256": "same",
                      "original_prompt": "What is 2+2?", "reasoning_available": True,
                      "final_available": True, "finish_reason": "stop"})
        log = EvalLog(status="success", eval=EvalSpec(
            created="2026-09-23T00:00:00Z", task="awareness", model="mockllm/model",
            dataset=EvalDataset(samples=1), config=EvalConfig(epochs=1),
            metadata={"response_mode": "auto"}), samples=[sample])
        path = tmp_path / f"{condition}.json"
        path.write_text(log.model_dump_json())
        paths.append(str(path))
    result = write_report(paths, tmp_path / "report")
    assert result["quality"]["baseline"]["reasoning_available"] == 1
    for name in ("summary.json", "samples.csv", "contrasts.csv", "audit.jsonl", "rates.png", "effects.png", "report.md"):
        assert (tmp_path / "report" / name).stat().st_size > 0
