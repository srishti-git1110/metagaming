"""Add the API judges (honesty_judge, cot_awareness) to mask_steer logs that were run with -T judge=false.

Runs on a laptop with OPENROUTER_API_KEY in .env; the pod never needs the key. Logs are rewritten in place with the new
scores appended (existing scores such as cot_words are kept). Logs that already have honesty_judge are skipped.

  python -m metagaming.steer.rescore results/steer-qwen36-27b/logs-*
"""
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import score
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metagaming.inspect_tasks.mask_gap import JUDGE_MODEL, honesty_judge  # noqa: E402
from metagaming.inspect_tasks.mask_probe import cot_awareness  # noqa: E402


def main(dirs, cot=True, scorers=None):
    for d in dirs:
        for info in list_eval_logs(str(d)):
            log = read_eval_log(info.name)
            if log.status != 'success':
                print('skip (status', log.status + ')', info.name)
                continue
            have = set((log.samples[0].scores or {}) if log.samples else {})
            if 'honesty_judge' in have:
                print('skip (already judged)', info.name)
                continue
            sc = scorers or ([honesty_judge(), cot_awareness()] if cot else [honesty_judge()])
            scored = score(log, sc, action='append', display='plain', model=JUDGE_MODEL)   # model= stops Inspect loading the steered HF model
            write_eval_log(scored, info.name)
            print('scored', info.name, 'samples', len(scored.samples or []))


if __name__ == '__main__':
    main(sys.argv[1:])
