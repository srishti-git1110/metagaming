"""Build steering directions (difference of means per layer) for one model and save them.

Directions (each saved twice, read at the last prompt token `:last` and as the mean over prompt tokens `:mean`):
  eval_generic   our 24 "you are being evaluated" vs "you are talking to a real user" system sentences
                 (metagaming/steer/contrasts.py), each followed by a neutral user turn
  graded         12 "your reply is scored" vs "nobody scores this" sentences, same layout
  rendering      the MASK provided_facts prompts themselves: benchmark form (E) minus real-use rewrite (D), on the
                 TRAIN half of the items (even positions in the sorted item list); the other half is held out
  benchmark_q    12 "this is a benchmark item" vs "this is a message from a person" sentences (item recognition
                 without any mention of the model being evaluated)
  eval_vs_neutral, deploy_vs_neutral   each side of eval_generic minus 24 neutral formatting sentences, so
                 "evaluation" and "deployment" are separate vectors (with eval_generic they are one line, v and -v)
  control_weather, control_book   unrelated-content pairs (sunny/rainy, book/film), same layout: semantic controls
  graded_orth    graded:last with its component along eval_generic:last removed (per layer), so "scored" is read with
                 the shared "this is not a normal conversation" part taken out (Flifenstein's point on Aranguri's post)
  random_k       Gaussian directions with the same per-layer norm as eval_generic:last (controls), k = 0..2
Also saved in the meta: residual_norm[layer] = mean last-token residual norm on the neutral prompts, so a steering
coefficient can be given as a fraction of the residual norm (steer_provider rel=true), the convention of the Claude
system cards and Read et al.

  python -m metagaming.steer.build_vectors --model Qwen/Qwen3.6-27B --out results/steer-qwen36-27b/vectors.pt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .contrasts import EVAL_GENERIC, NEUTRAL, NEUTRAL_USERS, SETS
from .hooks import load_model, render_chat, residuals, save_vectors

REPO = Path(__file__).resolve().parents[2]
RENDERINGS = REPO / 'results/mask-full-2026-11-06/renderings.jsonl'


def mask_items(renderings=RENDERINGS, config='provided_facts'):
    """{item: {'E': messages, 'D': messages}} for items that have both arms."""
    rows = [json.loads(l) for l in Path(renderings).read_text().splitlines()]
    by = defaultdict(dict)
    for r in rows:
        if r['pool'] == 'mask' and r['config'] == config and r['arm'] in ('E', 'D'):
            by[r['item']][r['arm']] = r['messages']
    return {it: arms for it, arms in sorted(by.items()) if {'E', 'D'} <= set(arms)}


def split_items(items, which):
    ids = sorted(items)
    return [i for k, i in enumerate(ids) if (k % 2 == 0) == (which == 'train')]


def diff_of_means(model, tok, pos_texts, neg_texts, batch_size):
    pos = residuals(model, tok, pos_texts, batch_size)
    neg = residuals(model, tok, neg_texts, batch_size)
    out = {}
    for key in ('last', 'mean'):
        p = torch.stack([x[key] for x in pos]).mean(0)
        n = torch.stack([x[key] for x in neg]).mean(0)
        out[key] = p - n
    stats = {key: {'pos_norm': torch.stack([x[key] for x in pos]).norm(dim=-1).mean(0).tolist()} for key in ('last',)}
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--renderings', default=str(RENDERINGS))
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='cap on train items for the rendering direction (smoke tests)')
    ap.add_argument('--items-file', default='', help='JSON list of item ids to build the rendering direction from instead of the '
                    'train half (e.g. the items where the base run lied in D and not in E); the test half is still held out')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--enable-thinking', default=None, choices=[None, 'true', 'false'])
    a = ap.parse_args()
    thinking = None if a.enable_thinking is None else a.enable_thinking == 'true'

    model, tok = load_model(a.model, a.dtype)
    directions, meta = {}, {'model': a.model, 'sets': {}, 'enable_thinking': thinking}

    for name, pairs in SETS.items():
        pos = [render_chat(tok, [{'role': 'system', 'content': p}, {'role': 'user', 'content': u}], thinking) for p, _ in pairs for u in NEUTRAL_USERS]
        neg = [render_chat(tok, [{'role': 'system', 'content': n}, {'role': 'user', 'content': u}], thinking) for _, n in pairs for u in NEUTRAL_USERS]
        d, stats = diff_of_means(model, tok, pos, neg, a.batch_size)
        for key, v in d.items():
            directions[f'{name}:{key}'] = v
        meta['sets'][name] = {'n_pos': len(pos), 'n_neg': len(neg), 'residual_norm_last': stats['last']['pos_norm']}

    neutral = [render_chat(tok, [{'role': 'system', 'content': n}, {'role': 'user', 'content': u}], thinking) for n in NEUTRAL for u in NEUTRAL_USERS]
    neutral_res = residuals(model, tok, neutral, a.batch_size)
    meta['residual_norm'] = torch.stack([x['last'] for x in neutral_res]).norm(dim=-1).mean(0).tolist()
    for name, side in (('eval_vs_neutral', 0), ('deploy_vs_neutral', 1)):
        pos = [render_chat(tok, [{'role': 'system', 'content': pair[side]}, {'role': 'user', 'content': u}], thinking) for pair in EVAL_GENERIC for u in NEUTRAL_USERS]
        d, _ = diff_of_means(model, tok, pos, neutral, a.batch_size)
        for key, v in d.items():
            directions[f'{name}:{key}'] = v
        meta['sets'][name] = {'n_pos': len(pos), 'n_neg': len(neutral)}

    items = mask_items(a.renderings)
    train = split_items(items, 'train')
    if a.items_file:
        test = set(split_items(items, 'test'))
        train = [i for i in json.loads(Path(a.items_file).read_text()) if i in items and i not in test]
    if a.limit:
        train = train[:a.limit]
    e = [render_chat(tok, items[i]['E'], thinking) for i in train]
    dd = [render_chat(tok, items[i]['D'], thinking) for i in train]
    d, stats = diff_of_means(model, tok, e, dd, a.batch_size)
    for key, v in d.items():
        directions[f'rendering:{key}'] = v
    meta['sets']['rendering'] = {'n_pos': len(e), 'n_neg': len(dd), 'train_items': train, 'test_items': split_items(items, 'test'),
                                 'residual_norm_last': stats['last']['pos_norm']}

    gv, ev = directions['graded:last'], directions['eval_generic:last']
    proj = (gv * ev).sum(-1, keepdim=True) / (ev * ev).sum(-1, keepdim=True).clamp(min=1e-6)
    directions['graded_orth:last'] = gv - proj * ev

    g = torch.Generator().manual_seed(0)
    ref = directions['eval_generic:last']
    for k in range(3):
        r = torch.randn(ref.shape, generator=g)
        directions[f'random_{k}:last'] = r / r.norm(dim=-1, keepdim=True) * ref.norm(dim=-1, keepdim=True)

    meta['layers'] = ref.shape[0]
    meta['hidden'] = ref.shape[1]
    meta['norms'] = {k: v.norm(dim=-1).tolist() for k, v in directions.items()}
    save_vectors(a.out, directions, meta)
    print(f'saved {len(directions)} directions x {meta["layers"]} layers to {a.out}')


if __name__ == '__main__':
    main()
