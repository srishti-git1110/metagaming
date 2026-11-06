"""Turn mask_rewrite .eval logs into renderings.jsonl for the full MASK set (arms E, D, E_Rbad, E_Rgood, D_Rbad, D_Rgood).

Same layout as results/profile-2026-10-24/renderings.jsonl, so mask_gap / mask_probe read it with -T renderings=...
The scored-example block for the R arms reuses the two October held-out example items (examples.json), which are
excluded from the item set.

  python -m metagaming.inspect_tasks.build_mask_full results/mask-full-2026-11-06
"""
import json
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from ..profile import OUT as OCTOBER, example_block
from .mask_rewrite import _request_text, mask_rows


def main(out_dir):
    out_dir = Path(out_dir)
    rewrites, failed = {}, {}
    for info in list_eval_logs(str(out_dir / 'rewrite-logs')):
        log = read_eval_log(info.name)
        for s in log.samples or []:
            sc = s.scores.get('rewrite_valid') if s.scores else None
            if sc and sc.metadata.get('rewrite'):
                rw = dict(sc.metadata['rewrite'])
                rw['request'] = _request_text(rw['request'])
                if rw['request'] and all(isinstance(t.get('content'), str) for t in rw['turns']):
                    rewrites[s.metadata['item']] = rw
                else:
                    failed[s.metadata['item']] = 'request or turn content not a string'
            else:
                failed[s.metadata['item']] = (sc.explanation if sc else 'no score')
    pairs = json.load(open(OCTOBER / 'examples.json'))['mask']
    held = {r['id'] for r in mask_rows() if any(r['system'] == p[0] for p in pairs)}
    rends = []
    for it in mask_rows():
        if it['id'] in held:
            continue
        rw = rewrites.get(it['id'])
        for arm in ('E', 'D', 'E_Rbad', 'E_Rgood', 'D_Rbad', 'D_Rgood'):
            if arm.startswith('D') and not rw:
                continue
            sys_, usr, pre = (it['system'], it['user'], []) if arm.startswith('E') else (rw['system'], rw['request'], rw['turns'])
            if arm.endswith('_Rbad') or arm.endswith('_Rgood'):
                sys_ = example_block('mask', pairs, 'bad' if arm.endswith('_Rbad') else 'good') + '\n\n' + sys_
            msgs = [{'role': 'system', 'content': sys_}] + pre + [{'role': 'user', 'content': usr}]
            rends.append({'rid': f'{it["id"]}|{arm}', 'item': it['id'], 'pool': 'mask', 'config': it['config'], 'arm': arm,
                          'request': usr, 'proposition': it['proposition'], 'truth': it['truth'], 'messages': msgs})
    (out_dir / 'renderings.jsonl').write_text('\n'.join(json.dumps(r) for r in rends) + '\n')
    (out_dir / 'build-summary.json').write_text(json.dumps(
        {'items': len({r['item'] for r in rends}), 'with_rewrite': len(rewrites) - len(held & set(rewrites)),
         'failed': failed, 'held_out_examples': sorted(held), 'renderings': len(rends)}, indent=1))
    print(f'{len(rends)} renderings; {len(rewrites)} rewrites ok, {len(failed)} failed, {len(held)} held-out example items')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/mask-full-2026-11-06')
