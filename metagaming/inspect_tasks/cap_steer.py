"""Inspect task: capability check under steering ("brain damage" control, Lindsey's criterion 3).

100 GSM8K test problems (HF openai/gsm8k, main split) through the same steer provider and settings as mask_steer;
exact match on the final number, no judge model needed. Compare the accuracy of each steered run with the coef=0 run:
a direction whose behavioural effect comes with a large accuracy drop is breaking the model, not steering a concept.

  python -m metagaming.steer.run_eval --task cap_steer --model steer/Qwen/Qwen3.6-27B -M ... --log-dir .../cap-<tag>
"""
import re
import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import metagaming.inspect_tasks.steer_provider  # noqa: E402,F401

NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def record_to_sample(rec):
    answer = rec['answer'].split('####')[-1].strip().replace(',', '')
    return Sample(input=rec['question'] + '\n\nGive the final numeric answer on the last line as "Answer: <number>".',
                  target=answer, metadata={'arm': 'gsm8k'})


def last_number(text):
    nums = NUM.findall(text or '')
    return nums[-1].replace(',', '') if nums else ''


@scorer(metrics=[accuracy(), stderr()])
def final_number():
    async def score(state: TaskState, target: Target) -> Score:
        got = last_number(state.output.completion)
        try:
            ok = abs(float(got) - float(target.text)) < 1e-6
        except ValueError:
            ok = False
        return Score(value=1.0 if ok else 0.0, answer=got, metadata={'empty_final': not (state.output.completion or '').strip()})
    return score


@task
def cap_steer(limit: int = 100, max_tokens: int = 4000, seed: int = 0):
    ds = hf_dataset('openai/gsm8k', data_dir='main', split='test', sample_fields=record_to_sample, shuffle=True, seed=seed, limit=limit)
    return Task(dataset=ds, solver=[generate()], scorer=final_number(), config=GenerateConfig(max_tokens=max_tokens, temperature=1.0))
