"""Inspect task for Feifan's gpt-oss-120b organisms served on a pod: the MASK gap plus grader probes and CoT judge.

Same task as mask_probe (honesty judge, four grader-belief probes answered by the target, CoT awareness judge), but
this file first registers Feifan's `harmony/` model provider, which renders the exact training prompt format
(fixed date 2025-08-05, developer message, reasoning on) and returns the reasoning as a ContentReasoning block.
Set FEIFAN_REPO to the clone of https://github.com/feifanl/causes-of-model-metagaming and HARMONY_BASE_URL to the
vLLM/SGLang server. See docs/pod-runbook.md.

  inspect eval metagaming/inspect_tasks/mask_organism.py --model harmony/gpt-oss-120b-bf16 \
      -T renderings=$PWD/results/mask-full-2026-11-06/renderings.jsonl -T configs=provided_facts --epochs 2
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Inspect loads task files standalone
_feifan = Path(os.environ.get('FEIFAN_REPO', Path.home() / 'causes-of-model-metagaming'))
sys.path.insert(0, str(_feifan / 'model-organisms' / 'evals'))
import harmony_provider  # noqa: E402,F401  registers the harmony/ provider

from inspect_ai import task  # noqa: E402

from metagaming.inspect_tasks.mask_probe import mask_probe  # noqa: E402


@task
def mask_organism(arms: str = 'E,D', paired: bool = True, limit: int = 0, renderings: str = '', configs: str = '',
                  condition: str = 'none', reasoning_effort: str = 'medium', max_tokens: int = 8000,
                  probe_max_tokens: int = 4500):
    return mask_probe(arms, paired, limit, renderings, configs, condition, reasoning_effort, max_tokens, probe_max_tokens)
