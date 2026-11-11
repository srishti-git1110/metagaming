"""Summarise mask_steer runs: per run (direction, layer, coef) the lie rate per arm with 95% CI, the paired E-D gap,
the share of unparsed / empty replies (coherence check) and mean reply length. Writes summary.json and a dose plot.

  python -m metagaming.inspect_tasks.report_mask_steer results/steer-qwen36-27b
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from ..aware_bench import mean, wilson
from .report_mask_probe import paired


def summarise(root):
    runs = []
    for log_dir in sorted(Path(root).glob('logs*')):
        for info in list_eval_logs(str(log_dir)):
            log = read_eval_log(info.name)
            if log.status != 'success' or not log.samples:
                continue
            ma = log.eval.model_args or {}
            run = {'dir': log_dir.name, 'model': log.eval.model, 'direction': ma.get('direction'), 'layer': ma.get('layer'),
                   'coef': float(ma.get('coef', 0) or 0), 'rel': bool(ma.get('rel', False)), 'scope': ma.get('scope', 'all'),
                   'prefill': ma.get('prefill') or '', 'arms': {}}
            per_item = defaultdict(lambda: defaultdict(list))
            by_arm = defaultdict(list)
            for s in log.samples:
                h = s.scores.get('honesty_judge')
                c = s.scores.get('cot_awareness')
                w = s.scores.get('cot_words')
                arm = s.metadata['arm']
                r = {'lie': None if h is None or h.answer == 'UNPARSED' else float(h.value), 'label': h.answer if h else None,
                     'empty': bool(h and h.metadata.get('empty_reply')) or bool(w and w.metadata.get('empty_final')),
                     'chars': len(s.output.message.text if s.output.message else ''),
                     'ta': None if c is None or c.value < 0 else int(c.value),
                     'words': {k: int(v) for k, v in w.metadata.items() if isinstance(v, bool)} if w else None}
                by_arm[arm].append(r)
                if r['lie'] is not None:
                    per_item[arm][s.metadata['item']].append(r['lie'])
            for arm, rs in by_arm.items():
                lies = [r['lie'] for r in rs if r['lie'] is not None]
                tas = [r['ta'] for r in rs if r['ta'] is not None]
                run['arms'][arm] = {'n': len(rs), 'lie_rate': round(mean(lies), 3) if lies else None,
                                    'lie_ci95': [round(x, 3) for x in wilson(int(sum(lies)), len(lies))] if lies else None,
                                    'unparsed_or_empty': round(mean([int((r['lie'] is None and r['label'] is not None) or r['empty']) for r in rs]), 3),
                                    'mean_chars': int(mean([r['chars'] for r in rs])),
                                    'cot_aware_any': round(mean([int(t > 0) for t in tas]), 3) if tas else None,
                                    'words': {k: round(mean([r['words'][k] for r in rs if r['words']]), 3)
                                              for k in (rs[0]['words'] or {})} if rs[0]['words'] else None}
            if 'E' in per_item and 'D' in per_item:
                run['gap_D_minus_E'] = paired(per_item['D'], per_item['E'])
                if run['coef'] == 0 and not ma.get('prefill'):
                    gap_items = sorted(i for i in per_item['D'] if i in per_item['E'] and mean(per_item['D'][i]) > mean(per_item['E'][i]))
                    Path(root, 'gap-items.json').write_text(json.dumps(gap_items))
                    run['gap_items'] = len(gap_items)
            runs.append(run)
    return runs


def main(root):
    runs = summarise(root)
    Path(root, 'summary.json').write_text(json.dumps(runs, indent=1))
    nan = float('nan')
    v = lambda d, k: d.get(k) if d.get(k) is not None else nan
    print(f'{"direction":20s} {"L":>3s} {"coef":>6s} {"lie E":>7s} {"lie D":>7s} {"gap D-E":>9s} {"CI":>16s} {"bad%":>5s} {"chars":>6s} {"decep E/D":>10s}')
    for r in sorted(runs, key=lambda r: (r['prefill'], str(r['direction']), r['layer'] or 0, r['coef'])):
        e, d, g = r['arms'].get('E', {}), r['arms'].get('D', {}), r.get('gap_D_minus_E') or {}
        bad = max(e.get('unparsed_or_empty', 0), d.get('unparsed_or_empty', 0))
        name = f'prefill:{r["prefill"][:14]}' if r['prefill'] else str(r['direction'])
        dec = f'{v(e.get("words") or {}, "deception"):.2f}/{v(d.get("words") or {}, "deception"):.2f}'
        print(f'{name:20s} {str(r["layer"]):>3s} {r["coef"]:+6.2f} {v(e, "lie_rate"):7.2f} '
              f'{v(d, "lie_rate"):7.2f} {v(g, "diff"):+9.2f} {str(g.get("ci95", "")):>16s} '
              f'{100 * bad:5.0f} {d.get("mean_chars", 0):6d} {dec:>10s}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        dirs = sorted({r['direction'] for r in runs if r['direction']})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        base = [r for r in runs if r['coef'] == 0]
        for ax, arm in zip(axes, ('E', 'D')):
            for dname in dirs:
                pts = sorted([(r['coef'], r['arms'][arm]['lie_rate']) for r in runs + base if (r['direction'] == dname or r['coef'] == 0) and arm in r['arms'] and r['arms'][arm]['lie_rate'] is not None])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=dname)
            ax.set_title({'E': 'Benchmark form (E)', 'D': 'Real-use rewrite (D)'}[arm])
            ax.set_xlabel('steering coefficient (fraction of residual norm if rel, else raw vector units)')
            ax.axhline(0, lw=0)
        axes[0].set_ylabel('lie rate (MASK provided_facts, held-out items)')
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(root, 'dose-response.png'), dpi=150)
        print('wrote', Path(root, 'dose-response.png'))
    except Exception as ex:  # noqa: BLE001
        print('plot skipped:', ex)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/steer-qwen36-27b')
