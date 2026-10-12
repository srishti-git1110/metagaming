# Source data

Run `python -m metagaming.prepare` from the repository root. No model calls occur.
The downloader pins [arbdwj/VEA-through-training](https://github.com/arbdwj/VEA-through-training)
at commit `2c1379ee9648c16884bb1634d554a27154d7a01c` and verifies SHA-256 values
embedded in `metagaming/data.py` before accepting files.

Downloaded files:

- `bench_prompts.jsonl`: the authors' exact 344 selected benchmark prompts.
- `fortress_prompts.jsonl`: their 100 Fortress prompts, including risk metadata.
- `judge_vea.py`: source of the original VEA rubric, parsed with `ast.literal_eval`
  without importing or running the source.
- `manifest.json`: generated provenance, counts, and source checksums.