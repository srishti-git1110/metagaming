"""Read mask_gap .eval logs and print lie rate per arm and the paired E-D gap per model.

  python -m metagaming.inspect_tasks.report_mask_gap results/organisms-mask-2026-11-06/logs
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from ..aware_bench import mean, wilson


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
        for a, b in (('D', 'E'), ('E_Rbad', 'E_Rgood'), ('D_Rbad', 'D_Rgood'), ('E_Rbad', 'E'), ('D_Rbad', 'D')):
            common = set(per.get(a, {})) & set(per.get(b, {}))
            if common:
                diffs = [mean(per[a][i]) - mean(per[b][i]) for i in common]
                rep['gaps'][f'{a} minus {b}'] = {'points': round(100 * mean(diffs), 1), 'items': len(common)}
    print(json.dumps(out, indent=1))
    Path(log_dir, '..', 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/organisms-mask-2026-11-06/logs')
