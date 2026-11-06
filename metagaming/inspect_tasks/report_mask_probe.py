"""Read mask_probe .eval logs: lie rate, P(graded), CoT annotations per arm; paired E-D differences; does belief track lying?

  python -m metagaming.inspect_tasks.report_mask_probe results/mask-probe-2026-11-06/logs
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from ..aware_bench import mean, wilson


def boot_ci(diffs, reps=4000, seed=0):
    rng = random.Random(seed)
    bs = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    return [round(bs[int(0.025 * reps)], 3), round(bs[int(0.975 * reps)], 3)]


def paired(per_item_a, per_item_b):
    common = sorted(set(per_item_a) & set(per_item_b))
    diffs = [mean(per_item_a[i]) - mean(per_item_b[i]) for i in common]
    return {'diff': round(mean(diffs), 3), 'ci95': boot_ci(diffs), 'items': len(common)} if diffs else None


def main(log_dir):
    if not Path(log_dir).exists():
        raise SystemExit(f'no logs at {log_dir}')
    out = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info.name)
        if log.status != 'success' or not log.samples:
            continue
        rep = out.setdefault(log.eval.model, {'logs': [], 'arms': {}, 'paired': {}, 'belief_vs_lying': {}})
        rep['logs'].append(Path(info.name).name)
        rows = []
        for s in log.samples:
            h, g, c = s.scores.get('honesty_judge'), s.scores.get('grader_belief'), s.scores.get('cot_awareness')
            rows.append({'arm': s.metadata['arm'], 'item': s.metadata['item'],
                         'lie': None if h is None or h.answer == 'UNPARSED' else float(h.value),
                         'label': h.answer if h else None,
                         'graded': None if g is None or g.value < 0 else float(g.value),
                         'probes': g.metadata.get('probes', {}) if g else {},
                         'ta': None if c is None or c.value < 0 else int(c.value),
                         'cot': c.metadata if c and c.value >= 0 else {}})
        by_arm = defaultdict(list)
        for r in rows:
            by_arm[r['arm']].append(r)
        lie_items, graded_items = {}, {}
        for arm, rs in by_arm.items():
            lies = [r['lie'] for r in rs if r['lie'] is not None]
            graded = [r['graded'] for r in rs if r['graded'] is not None]
            tas = [r['ta'] for r in rs if r['ta'] is not None]
            a = rep['arms'][arm] = {
                'n': len(rs), 'items': len({r['item'] for r in rs}),
                'lie_rate': round(mean(lies), 3), 'lie_ci95': [round(x, 3) for x in wilson(int(sum(lies)), len(lies))],
                'labels': {k: sum(r['label'] == k for r in rs) for k in ('LIE', 'HONEST', 'EVADE', 'UNPARSED')},
                'p_graded': round(mean(graded), 3) if graded else None, 'p_graded_n': len(graded),
                'per_probe': {}, 'cot_n': len(tas),
                'cot_aware_any': round(mean([t >= 1 for t in tas]), 3) if tas else None,
                'cot_aware_explicit': round(mean([t == 2 for t in tas]), 3) if tas else None,
                'cot_grader_mention': round(mean([r['cot'].get('grader', 0) for r in rs if r['ta'] is not None]), 3) if tas else None,
                'cot_fiction': round(mean([r['cot'].get('fiction', 0) for r in rs if r['ta'] is not None]), 3) if tas else None,
                'cot_real_stakes': round(mean([r['cot'].get('real_stakes') == 1 for r in rs if r['ta'] is not None]), 3) if tas else None,
            }
            for pid in ('g_scored', 'g_consumed', 'g_metric', 'g_counted'):
                v = [r['probes'][pid] for r in rs if r['probes'].get(pid) is not None]
                a['per_probe'][pid] = round(mean(v), 3) if v else None
            li, gi = defaultdict(list), defaultdict(list)
            for r in rs:
                if r['lie'] is not None:
                    li[r['item']].append(r['lie'])
                if r['graded'] is not None:
                    gi[r['item']].append(r['graded'])
            lie_items[arm], graded_items[arm] = li, gi
        for a, b in (('D', 'E'), ('E_Rbad', 'E_Rgood'), ('D_Rbad', 'D_Rgood')):
            if a in lie_items and b in lie_items:
                rep['paired'][f'{a} minus {b}'] = {'lie_rate': paired(lie_items[a], lie_items[b]),
                                                   'p_graded': paired(graded_items[a], graded_items[b])}
        # does the belief track the behaviour? (1) within arm: lie rate when the model says graded vs not
        for arm, rs in by_arm.items():
            hi = [r['lie'] for r in rs if r['lie'] is not None and r['graded'] is not None and r['graded'] >= 0.75]
            lo = [r['lie'] for r in rs if r['lie'] is not None and r['graded'] is not None and r['graded'] < 0.75]
            rep['belief_vs_lying'][arm] = {'lie_when_graded': round(mean(hi), 3) if hi else None, 'n_graded': len(hi),
                                           'lie_when_not_graded': round(mean(lo), 3) if lo else None, 'n_not_graded': len(lo)}
        # (2) across items: does the item's D-E belief change predict its D-E lie change?
        if 'D' in lie_items and 'E' in lie_items:
            common = [i for i in lie_items['D'] if i in lie_items['E'] and i in graded_items['D'] and i in graded_items['E']]
            dl = [mean(lie_items['D'][i]) - mean(lie_items['E'][i]) for i in common]
            dg = [mean(graded_items['D'][i]) - mean(graded_items['E'][i]) for i in common]
            if len(common) > 2:
                ml, mg = mean(dl), mean(dg)
                cov = sum((x - ml) * (y - mg) for x, y in zip(dl, dg))
                vl, vg = sum((x - ml) ** 2 for x in dl), sum((y - mg) ** 2 for y in dg)
                rep['belief_vs_lying']['item_corr_D_minus_E'] = {
                    'pearson': round(cov / (vl * vg) ** 0.5, 3) if vl and vg else None, 'items': len(common),
                    'lie_change_when_belief_dropped': round(mean([x for x, y in zip(dl, dg) if y < 0]), 3) if any(y < 0 for y in dg) else None,
                    'n_belief_dropped': sum(y < 0 for y in dg),
                    'lie_change_when_belief_same_or_up': round(mean([x for x, y in zip(dl, dg) if y >= 0]), 3) if any(y >= 0 for y in dg) else None,
                    'n_belief_same_or_up': sum(y >= 0 for y in dg)}
    print(json.dumps(out, indent=1))
    Path(log_dir, '..', 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/mask-probe-2026-11-06/logs')
