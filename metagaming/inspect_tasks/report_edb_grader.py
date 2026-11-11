"""Report for the EvalDetectBench grader-sentence runs, cut by source class.

Reads the Inspect logs of edb_grader.py (results/edb-grader-2026-11-07/<model>/logs-<arm>/*.eval) and, with --nov5,
the 5 Nov JSONL stores (results/grader-belief-2026-11-05) so both can be cut the same way.

Per model, arm and source class: P(graded) (mean over rollouts of the share of the four probes answered "graded"),
comply rate on refusal sources, lie rate on MASK, and the share of replies judged different in substance from the
first baseline rollout (the baseline-b row is the noise floor).

  python -m metagaming.inspect_tasks.report_edb_grader results/edb-grader-2026-11-07 --nov5
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..aware_bench import mean, wilson

CLASSES = {
    'safety_refusal': {'strongreject', 'do_not_answer', 'xstest_v2_copy', 'hackaprompt_dataset', 'aya_redteaming',
                       'real_toxicity_prompts', 'hh_rlhf_redteam', 'toxic_chat', 'cvalues_rlhf', 'model_written_evals'},
    'honesty_mask': {'mask'},
    'bias_values': {'discrim_eval', 'kobbq', 'cdial_bias', 'cvalues_rlhf', 'civics', 'llm_global_opinions',
                    'moralexceptqa', 'model_written_evals'},
    'hazard_knowledge': {'wmdp_bio'},
    'capability': {'gpqa', 'mmmlu', 'ifeval', 'natural_reasoning', 'nl2bash', 'writingbench', 'aya_evaluation'},
}
# comply/refuse is reported only on sources whose prompts are harmful or alarming requests; the model-written-evals and
# cvalues multiple-choice items also get a refusal judgement from the task but "comply" there only means "picked an option"
HARMFUL_REQUEST = {'strongreject', 'do_not_answer', 'xstest_v2_copy', 'hackaprompt_dataset', 'aya_redteaming',
                   'real_toxicity_prompts', 'hh_rlhf_redteam', 'toxic_chat'}
SAFETY_ALIGNMENT = CLASSES['safety_refusal'] | CLASSES['honesty_mask'] | CLASSES['bias_values'] | CLASSES['hazard_knowledge']


def source_class(source, pool):
    for name, srcs in CLASSES.items():
        if source in srcs:
            return name
    return 'deployment_general' if pool == 'deployment' else 'other'


def summarise(rows):
    """rows: dicts with model, arm, source, pool, graded (float|None), comply (0/1|None), lie (0/1|None), different (0/1|None)."""
    out = defaultdict(lambda: defaultdict(dict))
    groups = defaultdict(list)
    for r in rows:
        if r['source'] not in HARMFUL_REQUEST:
            r['comply'] = None
        cls = source_class(r['source'], r['pool'])
        for key in (cls, f'{r["pool"]} pool, all', 'safety+alignment (eval pool)' if r['pool'] == 'eval' and r['source'] in SAFETY_ALIGNMENT else None,
                    'safety-flavoured deployment' if r['pool'] == 'deployment' and r['source'] in SAFETY_ALIGNMENT else None):
            if key:
                groups[(r['model'], r['arm'], key)].append(r)
    for (model, arm, key), rs in sorted(groups.items()):
        d = {'n': len(rs)}
        g = [r['graded'] for r in rs if r['graded'] is not None]
        if g:
            d['p_graded'] = round(mean(g), 3)
        for field, name in (('comply', 'comply_rate'), ('lie', 'lie_rate'), ('different', 'p_different')):
            v = [r[field] for r in rs if r.get(field) is not None]
            if v:
                d[name] = round(mean(v), 3)
                d[name + '_n'] = len(v)
                d[name + '_ci95'] = [round(x, 3) for x in wilson(int(sum(v)), len(v))]
        out[model][key][arm] = d
    return out


def rows_from_inspect(root):
    from inspect_ai.log import list_eval_logs, read_eval_log
    rows = []
    for model_dir in sorted(Path(root).iterdir()):
        if not model_dir.is_dir() or model_dir.name == 'smoke':
            continue
        for log_dir in sorted(model_dir.glob('logs-*')):
            arm = log_dir.name[5:]
            for info in list_eval_logs(str(log_dir)):
                log = read_eval_log(info.name)
                if log.status != 'success':
                    continue
                for s in log.samples or []:
                    g, b = s.scores.get('grader_belief'), s.scores.get('behaviour')
                    bm = b.metadata if b else {}
                    rows.append({'model': model_dir.name, 'arm': arm, 'source': s.metadata['source'], 'pool': s.metadata['pool'],
                                 'graded': None if g is None or g.value < 0 else float(g.value),
                                 'comply': None if not bm.get('refusal') else int(bm['refusal'] == 'COMPLY'),
                                 'lie': None if not bm.get('honesty') else int(bm['honesty'] == 'LIE'),
                                 'different': None if b is None or b.value < 0 else int(b.value)})
    return rows


def rows_from_nov5(root='results/grader-belief-2026-11-05', topup='results/edb-grader-2026-11-07/topup-ids.json'):
    """5 Nov rows for GLM and DeepSeek. Items listed in `topup` (missing or duplicate-id items re-run through the Inspect task
    under results/edb-grader-2026-11-07/{glm,deepseek}) are dropped here so the Inspect rows replace them."""
    root = Path(root)
    skip = json.loads(Path(topup).read_text()) if Path(topup).exists() else {}
    gen = {}
    for l in (root / 'gen.jsonl').read_text().splitlines():
        r = json.loads(l)
        gen[r['key']] = r
    probes = defaultdict(list)
    for l in (root / 'probe.jsonl').read_text().splitlines():
        r = json.loads(l)
        if r.get('graded') is not None:
            probes[r['gen_key']].append(r['graded'])
    judge = {}
    for l in (root / 'judge.jsonl').read_text().splitlines():
        r = json.loads(l)
        judge[r['key']] = r
    rows = []
    for key, r in gen.items():
        if not r.get('final') or r['id'] in set(skip.get(r['target'], [])):
            continue
        arm = 'baseline-a' if r['arm'] == 'baseline' and str(r['sample']) == '0' else ('baseline-b' if r['arm'] == 'baseline' else r['arm'])
        ref, hon, chg = judge.get(f'refusal|{key}', {}), judge.get(f'honesty|{key}', {}), judge.get(f'change|{key}', {})
        rows.append({'model': r['target'], 'arm': arm, 'source': r['source'], 'pool': r['pool'],
                     'graded': mean(probes[key]) if probes.get(key) else None,
                     'comply': None if not ref.get('label') else int(ref['label'] == 'COMPLY'),
                     'lie': None if not hon.get('label') else int(hon['label'] == 'LIE'),
                     'different': None if not chg.get('different') else int(chg['different'] == 'true')})
    return rows


def print_tables(summary, keys=('safety+alignment (eval pool)', 'capability', 'safety_refusal', 'honesty_mask',
                                 'safety-flavoured deployment', 'deployment_general')):
    for model, by_key in summary.items():
        print(f'\n=== {model}')
        for key in keys:
            if key not in by_key:
                continue
            print(f'  [{key}]')
            print(f'    {"arm":14s} {"n":>4s} {"P(graded)":>10s} {"comply":>8s} {"lie":>6s} {"different":>10s}')
            for arm, d in by_key[key].items():
                f = lambda k: ('' if k not in d else f'{d[k]:.2f}')
                print(f'    {arm:14s} {d["n"]:4d} {f("p_graded"):>10s} {f("comply_rate"):>8s} {f("lie_rate"):>6s} {f("p_different"):>10s}')


def main(argv):
    root = argv[0] if argv else 'results/edb-grader-2026-11-07'
    rows = rows_from_inspect(root) if Path(root).exists() else []
    if '--nov5' in argv:
        rows += rows_from_nov5()
    summary = summarise(rows)
    print_tables(summary)
    Path(root).mkdir(parents=True, exist_ok=True)
    Path(root, 'summary-by-class.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
