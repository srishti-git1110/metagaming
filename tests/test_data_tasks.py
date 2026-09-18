import hashlib

import pytest

from metagaming.conditions import CONDITIONS
from metagaming.data import checked_bytes, load_records, vea_rubric
from metagaming.tasks import awareness, build_samples


def test_original_sample_set_and_ids():
    rows = load_records()
    assert len(rows) == len({r["prompt_id"] for r in rows}) == 444
    assert sum(r["benchmark"] == "agentharm" for r in rows) == 44
    assert "{cot}" in vea_rubric()


def test_conditions_preserve_every_original_question():
    baseline = build_samples("baseline")
    for name, condition in CONDITIONS.items():
        changed = build_samples(name)
        for original, modified in zip(baseline, changed, strict=True):
            assert original.id == modified.id
            assert original.input[-1].role == modified.input[-1].role
            assert original.input[-1].content == modified.input[-1].content
            assert original.metadata["prompt_sha256"] == modified.metadata["prompt_sha256"]
            assert len(original.input) == 1
            assert original.input[0].role == "user"
            if condition.system:
                assert len(modified.input) == 2
                assert modified.input[0].role == "system"
                assert modified.input[0].content == condition.system


def test_construct_tasks_without_model_access():
    from evals import awareness as cli_task
    assert len(cli_task("mockllm/model").dataset) == 444
    for condition in CONDITIONS:
        task = awareness("mockllm/model", condition=condition)
        assert len(task.dataset) == 444
        assert task.config.temperature == .6
        assert task.config.top_p == .95
    assert len(build_samples("baseline", "fortress")) == 100


@pytest.mark.parametrize("kwargs", [
    {"condition": "typo"}, {"benchmark": "typo"}, {"response_mode": "typo"},
    {"epochs": 0}, {"max_tokens": 0},
])
def test_reject_invalid_config(kwargs):
    with pytest.raises(ValueError):
        awareness("mockllm/model", **kwargs)


def test_missing_or_changed_data_fails(tmp_path):
    path = tmp_path / "data.jsonl"
    digest = hashlib.sha256(b"original").hexdigest()
    with pytest.raises(FileNotFoundError, match="prepare"):
        checked_bytes(path, digest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        checked_bytes(path, digest)
