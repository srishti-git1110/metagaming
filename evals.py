"""Inspect CLI entry point: inspect eval evals.py@awareness ..."""
from inspect_ai import Task, task
from metagaming.tasks import awareness as build_awareness


@task
def awareness(
    judge_model: str,
    condition: str = "baseline",
    benchmark: str = "all",
    response_mode: str = "olmo_think",
    epochs: int = 8,
    max_tokens: int = 8192,
) -> Task:
    """Original safety prompts with a selectable evaluation-context intervention."""
    return build_awareness(judge_model, condition, benchmark, response_mode, epochs, max_tokens)
