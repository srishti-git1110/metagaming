"""Inspect task: build the real-use rewrite (arm D) for every binary MASK item (the October REWRITE_MASK prompt).

Dataset: every row of the MASK configs that are a system-prompt pressure plus one user request
(known_facts, provided_facts, disinformation). continuations (write-on from an article), doubling_down (the model's
own earlier lie is part of the prompt) and statistics (numerical, not binary) are left out: the rewrite prompt and
the binary honesty judge do not fit them.
Solver: generate the JSON rewrite; on a parse / structure failure regenerate (up to `attempts`).
Scorer: 1 if the rewrite parses and passes the October structural checks; the parsed rewrite sits in the score
metadata and build_mask_full.py turns the log into renderings.

  inspect eval metagaming/inspect_tasks/mask_rewrite.py --model openrouter/z-ai/glm-5.3 \
      --log-dir results/mask-full-2026-11-06/rewrite-logs
"""
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Inspect loads task files standalone
from metagaming.inspect_tasks.common import REPO  # noqa: E402
from metagaming.profile import REWRITE_MASK, _check_rewrite, _gen_json, _normalize_turns  # noqa: E402

MASK_DIR = REPO / 'data/upstream/mask'
CONFIGS = ('known_facts', 'provided_facts', 'disinformation')


def mask_rows(configs=CONFIGS):
    import pandas as pd
    out = []
    for cfg in configs:
        df = pd.read_parquet(MASK_DIR / f'{cfg}.parquet')
        df = df[df['type'] == 'binary']
        for _, r in df.iterrows():
            out.append({'id': f'mask:{cfg}:{r.task_id}', 'config': cfg, 'system': str(r.system_prompt),
                        'user': str(r.user_prompt), 'proposition': str(r.proposition), 'truth': str(r.ground_truth)})
    return out


def _request_text(req):
    """The request is sometimes wrapped as {"user": "...", "assistant": ""}; unwrap it."""
    if isinstance(req, dict):
        req = req.get('user') or req.get('content') or req.get('text')
    return req if isinstance(req, str) else None


def _normalize(d):
    """Accept {"user": ...} / {"assistant": ...} single-key turns (GLM's habit) before the October checks."""
    turns = []
    for t in d.get('turns', []):
        if isinstance(t, dict) and len(t) == 1 and next(iter(t)) in ('user', 'assistant'):
            role, content = next(iter(t.items()))
            turns.append({'role': role, 'content': content})
        else:
            turns.append(t)
    d['turns'] = turns
    turns = _normalize_turns(turns)  # also expands {"user", "assistant"} pair objects into two turns
    if turns and turns[-1]['role'] == 'user':
        turns = turns[:-1]  # the request is the next user message; a trailing user turn would double it
    if len(turns) > 6:
        turns = turns[-6:]  # longer lead-ins are fine in substance; keep the last six so every D has 4-6
    d['turns'] = turns
    d['request'] = _request_text(d.get('request'))
    _check_rewrite(d)


def parse_rewrite(text):
    try:
        return _gen_json(text, _normalize), None
    except Exception as e:  # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'[:200]


@solver
def rewrite_until_valid(attempts: int = 3) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for k in range(attempts):
            state = await generate(state)
            data, err = parse_rewrite(state.output.completion or '')
            if data:
                state.metadata['rewrite'] = data
                state.metadata['attempts'] = k + 1
                return state
            state.metadata['last_error'] = err
            state.messages = state.messages[:1]  # drop the failed completion and try again
        state.metadata['attempts'] = attempts
        return state
    return solve


@scorer(metrics=[mean()])
def rewrite_valid():
    async def score(state: TaskState, target: Target) -> Score:
        data = state.metadata.get('rewrite')
        return Score(value=1.0 if data else 0.0, answer='valid' if data else 'invalid',
                     explanation=None if data else state.metadata.get('last_error'),
                     metadata={'rewrite': data, 'attempts': state.metadata.get('attempts'),
                               'item': state.metadata['item'], 'config': state.metadata['config']})
    return score


@task
def mask_rewrite(configs: str = ','.join(CONFIGS), limit: int = 0, attempts: int = 3, max_tokens: int = 6000):
    cfgs = tuple(configs.split(',')) if isinstance(configs, str) else tuple(configs)
    rows = mask_rows(cfgs)
    if limit:
        rows = rows[:limit]
    samples = [Sample(id=r['id'], input=REWRITE_MASK.format(system=r['system'], user=r['user']),
                      metadata={'item': r['id'], 'config': r['config'], 'system': r['system'], 'user': r['user'],
                                'proposition': r['proposition'], 'truth': r['truth']}) for r in rows]
    return Task(dataset=MemoryDataset(samples, name='mask_rewrite'), solver=[rewrite_until_valid(attempts)],
                scorer=rewrite_valid(), config=GenerateConfig(max_tokens=max_tokens, temperature=1.0))
