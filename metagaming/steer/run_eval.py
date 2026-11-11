"""Run the mask_steer (or, with --task cap_steer, the capability-check) task through Inspect's Python API with the `steer` provider registered first
(the `inspect eval` CLI resolves the model before it loads the task file, so the provider is unknown to it).

  python -m metagaming.steer.run_eval --model steer/Qwen/Qwen3.6-27B --log-dir results/steer-qwen36-27b/logs-base \
      -M vectors=results/steer-qwen36-27b/vectors.pt -M direction=eval_generic:last -M layer=24 -M coef=0 \
      -M auto_model_class=AutoModelForImageTextToText -M dtype=bfloat16 -M enable_thinking=true \
      -T split=test --epochs 2 --max-connections 16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import metagaming.inspect_tasks.steer_provider  # noqa: E402,F401
from inspect_ai import eval as inspect_eval  # noqa: E402
from metagaming.inspect_tasks.cap_steer import cap_steer  # noqa: E402
from metagaming.inspect_tasks.mask_steer import mask_steer  # noqa: E402

TASKS = {'mask_steer': mask_steer, 'cap_steer': cap_steer}


def parse_kv(items):
    out = {}
    for kv in items or []:
        k, v = kv.split('=', 1)
        if v.lower() in ('true', 'false'):
            v = v.lower() == 'true'
        else:
            try:
                v = int(v) if v.lstrip('+-').isdigit() else float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='mask_steer', choices=sorted(TASKS))
    ap.add_argument('--model', required=True)
    ap.add_argument('--log-dir', required=True)
    ap.add_argument('-M', action='append', default=[], help='model arg key=value')
    ap.add_argument('-T', action='append', default=[], help='task arg key=value')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--max-connections', type=int, default=16, help='= generation batch size')
    ap.add_argument('--display', default='plain')
    a = ap.parse_args()
    logs = inspect_eval(TASKS[a.task](**parse_kv(a.T)), model=a.model, model_args=parse_kv(a.M), log_dir=a.log_dir,
                        epochs=a.epochs, max_connections=a.max_connections, display=a.display)
    for log in logs:
        print(log.location, log.status)
    sys.exit(0 if all(l.status == 'success' for l in logs) else 1)


if __name__ == '__main__':
    main()
