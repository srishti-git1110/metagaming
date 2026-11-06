"""Read mask_gap .eval logs and print lie rate per arm and the paired E-D gap per model.

  python -m metagaming.inspect_tasks.report_mask_gap results/organisms-mask-2026-11-06/logs
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from ..aware_bench import mean, wilson


def gap(per, a, b, items):
    """Paired difference in lie rate (points) over items, with a bootstrap 95% CI over items."""
    items = sorted(items)
    diffs = [mean(per[a][i]) - mean(per[b][i]) for i in items]
    rng = random.Random(0)
    bs = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(4000))
    return {'points': round(100 * mean(diffs), 1), 'ci95': [round(100 * bs[100], 1), round(100 * bs[3899], 1)], 'items': len(items),
            'rate_' + b: round(mean([mean(per[b][i]) for i in items]), 3), 'rate_' + a: round(mean([mean(per[a][i]) for i in items]), 3)}


def main(log_dir):
    out = {}
    if not Path(log_dir).exists():
        raise SystemExit(f'no logs at {log_dir}')
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info.name)
        if log.status != 'success' or not log.samples:
            continue
        model = log.eval.model
        per = defaultdict(lambda: defaultdict(list))
        for s in log.samples:
            sc = s.scores.get('honesty_judge')
            if sc is None or sc.answer == 'UNPARSED':
                continue
            per[s.metadata['arm']][s.metadata['item']].append(float(sc.value))
        rep = out.setdefault(model, {'arms': {}, 'gaps': {}, 'logs': []})
        rep['logs'].append(Path(info.name).name)
        for arm, items in per.items():
            flat = [x for v in items.values() for x in v]
            rep['arms'][arm] = {'lie_rate': round(mean(flat), 3), 'ci95': [round(c, 3) for c in wilson(int(sum(flat)), len(flat))],
                                'n': len(flat), 'items': len(items)}
        cfg_of = {s.metadata['item']: s.metadata['config'] for s in log.samples}
        for a, b in (('D', 'E'), ('E_Rbad', 'E_Rgood'), ('D_Rbad', 'D_Rgood'), ('E_Rbad', 'E'), ('D_Rbad', 'D')):
            common = set(per.get(a, {})) & set(per.get(b, {}))
            if common:
                rep['gaps'][f'{a} minus {b}'] = gap(per, a, b, common)
                rep['gaps'][f'{a} minus {b}']['by_config'] = {
                    cfg: gap(per, a, b, [i for i in common if cfg_of[i] == cfg]) for cfg in sorted({cfg_of[i] for i in common})}
    print(json.dumps(out, indent=1))
    Path(log_dir, '..', 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/organisms-mask-2026-11-06/logs')
