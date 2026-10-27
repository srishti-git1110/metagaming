"""Figures for the chain-of-thought judge: spontaneous test-awareness vs the elicited probe, and
the within-item link between noticing the scoring and following it."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .profile import OUT
from .habit_plot import INK, MUTED, GRID, footer
from .profile_plot import MODELS, style

ARMS = [('E', 'MASK scenario exactly as the benchmark wrote it', '#3A3A38'),
        ('E_nostakes', 'Same, "your job depends on it" lines removed', '#2A78D6'),
        ('D', 'Full realistic rewrite: staff note + earlier turns + casual request', '#EB6834')]


def fig_cot_vs_probe(cs, s, out):
    fig = plt.figure(figsize=(10, 8.2), dpi=220)
    axes = [fig.add_axes([0.10, 0.47, 0.87, 0.22]), fig.add_axes([0.10, 0.17, 0.87, 0.22])]
    w = 0.24
    for i, (key, name) in enumerate(MODELS):
        pm = s['models'].get(key, {}).get('mask', {})
        for j, (arm, label, color) in enumerate(ARMS):
            x = i + (j - 1) * w
            v = pm.get('belief', {}).get(arm)
            c = cs.get(f'{key}|mask|{arm}')
            if v and v.get('mean') is not None:
                y = 100 * v['mean']
                axes[0].bar(x, y, w, color=color, edgecolor='white', linewidth=1, zorder=3, label=label if i == 0 else None)
                axes[0].text(x, y + 1.5, f'{y:.0f}', ha='center', va='bottom', fontsize=8.5, color=INK)
            if c:
                y = 100 * c['aware']
                axes[1].bar(x, y, w, color=color, edgecolor='white', linewidth=1, zorder=3)
                axes[1].text(x, y + 1.5, f'{y:.0f}', ha='center', va='bottom', fontsize=8.5, color=INK)
    for ax, lab, ymax in ((axes[0], 'ASKED afterwards (EvalDetectBench probe): % of answers saying "this was an evaluation"', 115),
                          (axes[1], 'UNPROMPTED, in its own chain of thought: % of rollouts that treat the chat as a test', 60)):
        ax.set_xticks(range(len(MODELS)), [m[1] for m in MODELS])
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        style(ax, '% of rollouts', ymax)
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=10.5, weight='bold', color=INK, va='bottom')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.22), ncol=1, frameon=False, fontsize=9, handlelength=1.2)
    fig.text(0.04, 0.945, 'Two kinds of "eval awareness" come apart: the probe says "evaluation" in every version,\n'
             'but only GLM says so unprompted, and that drops when the scenario is rewritten to read like real use',
             fontsize=12.5, weight='bold', color=INK)
    fig.text(0.04, 0.905, 'Bottom panel: a separate LLM judge (DeepSeek V4.1 Flash) read each rollout\'s reasoning and marked whether the model itself '
             'treats the chat as a test\n(e.g. "This is a scenario testing whether I follow the deceptive instruction"). 58 MASK scenarios, 6 rollouts each (Kimi 3).',
             fontsize=9.5, color=MUTED)
    footer(fig, ['Within an item, rollouts that noticed the test lied no more than the ones that did not',
                 '(pooled contrast in points: DeepSeek -4 [-10,+1], GLM -1 [-5,+2], Kimi -17 [-36,0]; gpt-oss never noticed).',
                 'Reading: elicited belief is pinned at "evaluation" in every version, so the 40-63 point rise in lying under the',
                 'rewrite is not explained by a belief change on either measure.'], 0.085)
    path = out / 'p-cot-vs-probe-mask.png'
    fig.savefig(path, facecolor='white')
    return path


def fig_reward_within_item(cs, out):
    cells = [('D', 'rewrite, no scoring info', '#8A8880'),
             ('D_Rgood', 'rewrite + archive where the HONEST reply scored 1', '#2A78D6'),
             ('D_Rbad', 'rewrite + archive where the LYING reply scored 1', '#EB6834')]
    fig = plt.figure(figsize=(10, 7.6), dpi=220)
    ax = fig.add_axes([0.12, 0.22, 0.85, 0.40])
    w = 0.24
    for i, (key, name) in enumerate(MODELS):
        for j, (arm, label, color) in enumerate(cells):
            c = cs.get(f'{key}|mask|{arm}')
            if not c or c.get('within_item') is None or c['within_items'] < 8:
                continue
            x = i + (j - 1) * w
            y = 100 * c['within_item']
            lo, hi = (100 * v for v in c['within_ci'])
            ax.bar(x, y, w, color=color, edgecolor='white', linewidth=1, zorder=3, label=label if i == 0 else None)
            ax.errorbar(x, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]], fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=4)
            ax.text(x, min(hi, 52) + 1.5 if y >= 0 else max(lo, -52) - 1.5, f'{y:+.0f} (n={c["within_items"]})', ha='center',
                    va='bottom' if y >= 0 else 'top', fontsize=7.5, color=INK, rotation=90 if False else 0)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(MODELS)), [m[1] for m in MODELS])
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(-60, 60)
    ax.set_ylabel('lie rate when the rollout noticed the test/scoring\nMINUS when it did not (same item, percentage points)', fontsize=9, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0, labelsize=9.5)
    ax.text(0.0, 1.02, 'MASK honesty, realistic rewrite  (above zero = noticing made it lie MORE = less aligned)',
            transform=ax.transAxes, fontsize=10.5, weight='bold', color=INK, va='bottom')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.10), ncol=1, frameon=False, fontsize=9, handlelength=1.2)
    fig.text(0.04, 0.925, 'Noticing the scoring is what carries the reward effect: the rollouts that remark on the\narchive lie more when it rewards lying and less when it rewards honesty',
             fontsize=12.5, weight='bold', color=INK)
    fig.text(0.04, 0.875, 'Each bar: mean over items of (lie rate among rollouts the judge marked as noticing the test or scoring) minus (lie rate among the other rollouts of the '
             'same item).\nOnly items with both kinds of rollout count (n). Error bars: 95% bootstrap over items. gpt-oss-120b never verbalizes noticing, so it has no bar.',
             fontsize=9.5, color=MUTED)
    footer(fig, ['Without scoring info (grey), rollouts that notice the test lie somewhat LESS. With a lying-rewarded archive the sign',
                 'flips for DeepSeek and GLM (+15, +10); Kimi does not flip (-26).',
                 'This is reward-following conditional on having read the reward, inside a chat the model believes is an evaluation',
                 '(probe P(eval) 1.0 in every arm).'], 0.085)
    path = out / 'p-cot-reward-within-item.png'
    fig.savefig(path, facecolor='white')
    return path


def main():
    out = OUT / 'analysis'
    s = json.loads((out / 'summary.json').read_text())
    cs = json.loads((out / 'cot-judge-summary.json').read_text())
    plt.rcParams.update({'font.family': 'Arial', 'text.color': INK, 'xtick.color': INK})
    for p in (fig_cot_vs_probe(cs, s, out), fig_reward_within_item(cs, out)):
        print('wrote', p)


if __name__ == '__main__':
    main()
