"""No generation occurs when task definitions are constructed or imported."""
import hashlib

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import solver

from .conditions import CONDITIONS
from .data import COMMIT, COUNTS, load_records
from .scoring import refusal, split_response, vea


@solver
def sample_response(response_mode: str, seed: int = 42):
    async def solve(state, generate):
        rollout_seed = seed + state.epoch - 1
        state = await generate(state, seed=rollout_seed)
        parts = split_response(state.output, response_mode)
        state.metadata.update(
            rollout_seed=rollout_seed, reasoning_source=parts.reasoning_source,
            think_closed=parts.closed, finish_reason=state.output.stop_reason,
            final_available=bool(parts.final), reasoning_available=bool(parts.reasoning),
        )
        return state
    return solve


def build_samples(condition: str, benchmark: str = "all") -> list[Sample]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}. Choose from {list(CONDITIONS)}")
    if benchmark != "all" and benchmark not in COUNTS:
        raise ValueError(f"Unknown benchmark {benchmark!r}. Choose all or {list(COUNTS)}")
    intervention = CONDITIONS[condition]
    samples = []
    for row in load_records():
        if benchmark != "all" and row["benchmark"] != benchmark:
            continue
        messages = []
        if intervention.system is not None:
            messages.append(ChatMessageSystem(content=intervention.system))
        messages.append(ChatMessageUser(content=row["prompt"]))
        samples.append(Sample(
            id=row["prompt_id"], input=messages,
            metadata={
                **{k: v for k, v in row.items() if k != "prompt"},
                "original_prompt": row["prompt"], "condition": condition,
                "family": intervention.family, "cue": intervention.cue,
                "grading": intervention.grading, "source_commit": COMMIT,
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            },
        ))
    return samples


def awareness(
    judge_model: str,
    condition: str = "baseline",
    benchmark: str = "all",
    response_mode: str = "olmo_think",
    epochs: int = 8,
    max_tokens: int = 8192,
) -> Task:
    """
    response_mode=olmo_think requires a backend that begins in a think block.
    response_mode=auto handles provider reasoning or explicit think tags and marks
    unavailable reasoning as missing rather than pretending VEA is zero.
    """
    if not judge_model.strip():
        raise ValueError("An explicit judge_model is required")
    if response_mode not in {"olmo_think", "auto"}:
        raise ValueError("response_mode must be olmo_think or auto")
    if epochs < 1 or max_tokens < 1:
        raise ValueError("epochs and max_tokens must be positive")
    return Task(
        dataset=MemoryDataset(build_samples(condition, benchmark), name="olmo_444"),
        solver=sample_response(response_mode),
        scorer=[vea(judge_model, response_mode), refusal(judge_model, response_mode)],
        epochs=epochs,
        config=GenerateConfig(temperature=0.6, top_p=0.95, max_tokens=max_tokens, seed=42),
        metadata={
            "condition": condition, "benchmark": benchmark, "source_commit": COMMIT,
            "response_mode": response_mode, "judge_model": judge_model,
            "refusal_rubric": "adapted_f2_v1", "vea_rubric": "upstream_f1",
            "inference_parity": "not_exact_see_docs",
        },
    )
