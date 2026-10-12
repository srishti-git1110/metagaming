"""Pinned prompt acquisition. Importing this module never downloads anything."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "upstream"
COMMIT = "2c1379ee9648c16884bb1634d554a27154d7a01c"
BASE_URL = f"https://raw.githubusercontent.com/arbdwj/VEA-through-training/{COMMIT}/"
FILES = {
    "data/prompts/bench_prompts.jsonl": "fd6b41584450dd10e6b0e992a912ab09a6b9575c26519d0ee2674c351598dff5",
    "data/prompts/fortress_prompts.jsonl": "7aebff23ae9e0479e5e51a01a5c28bb2b0fb6dc9fc2937d0f07d3d20c8eef80d",
    "pipeline/judge_vea.py": "e206482e3e034a80321a86cb743d827bf2d943f57721da386c4eb0df7f189f41",
}
COUNTS = {"jbb": 100, "advbench": 100, "harmfulqa": 100, "agentharm": 44, "fortress": 100}


def checked_bytes(path: Path, expected: str) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python -m metagaming.prepare")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return content


def prepare(directory: Path = DATA_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for source, digest in FILES.items():
        destination = directory / Path(source).name
        if destination.exists():
            checked_bytes(destination, digest)
            continue
        with urlopen(BASE_URL + source, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Upstream checksum mismatch: {source}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
    load_records(directory)
    (directory / "manifest.json").write_text(json.dumps({
        "repository": "https://github.com/arbdwj/VEA-through-training",
        "commit": COMMIT, "sha256": FILES, "counts": COUNTS,
    }, indent=2) + "\n")


def load_records(directory: Path = DATA_DIR) -> list[dict]:
    rows = []
    for source, digest in FILES.items():
        if not source.endswith(".jsonl"):
            continue
        content = checked_bytes(directory / Path(source).name, digest)
        for line in content.decode().splitlines():
            row = json.loads(line)
            if "fortress" in source:
                row["prompt_id"] = f"fortress_{row['prompt_id']}"
                row["benchmark"] = "fortress"
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError("Empty or invalid source prompt")
            rows.append(row)
    if Counter(r["benchmark"] for r in rows) != COUNTS:
        raise ValueError("Prompt counts differ from the paper's fixed 444-prompt set")
    if len({r["prompt_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prompt IDs")
    return rows


def vea_rubric(directory: Path = DATA_DIR) -> str:
    source = "pipeline/judge_vea.py"
    content = checked_bytes(directory / "judge_vea.py", FILES[source])
    # Parse a literal; never import or execute downloaded Python.
    for node in ast.parse(content).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "JUDGE_PROMPT" for t in node.targets
        ):
            prompt = ast.literal_eval(node.value)
            if not isinstance(prompt, str) or "{cot}" not in prompt:
                break
            return prompt
    raise ValueError("Pinned source does not contain the expected judge rubric")
