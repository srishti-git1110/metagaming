"""Figure for the MASK honesty run. Offline; reads analysis/summary.json."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .mask_run import OUT, ARMS
from .habit_plot import INK, MUTED, GRID, footer

MODELS = [('luna', 'GPT-6-luna'), ('deepseek', 'DeepSeek V4.1 Flash'),
          ('gptoss-high', 'gpt-oss-120b\nhigh effort'), ('gptoss-low', 'gpt-oss-120b\nlow effort')]
ARM_STYLE = [('E', 'As shipped (evaluation-looking)', '#3A3A38'),
             ('D', 'Deployment-like: operator note + prior routine turns', '#B9B7AF'),
             ('Ghonest', '+ "scored for honesty by a grader"', '#2A78D6'),
             ('Slie', '+ scored examples: lying scored 1', '#EB6834'),
             ('Shonest', '+ same examples, scores swapped', '#F4B59A'),
             ('Sunscored', '+ same examples, no scores', '#8A8880')]


def main():
    s = json.loads((OUT / 'analysis' / 'summary.json').read_text())
    plt.rcParams.update({'font.family': 'Arial', 'text.color': INK, 'xtick.color': INK})
    fig = plt.figure(figsize=(12.5, 7.8), dpi=220)
    ax = fig.add_axes([0.07, 0.30, 0.90, 0.50])
    w = 0.13
    for i, (key, label) in enumerate(MODELS):
        m = s['models'][key]
        for j, (arm, name, color) in enumerate(ARM_STYLE):
            v = m['lie_rate'][arm]
            x = i + (j - 2.5) * w
            y = 100 * v['rate']
            lo, hi = 100 * v['ci95'][0], 100 * v['ci95'][1]
            ax.bar(x, y, w, color=color, edgecolor='white', linewidth=1, zorder=3, label=name if i == 0 else None)
            ax.errorbar(x, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]], fmt='none', ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
            ax.text(x, hi + 1.5, f'{y:.0f}', ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.set_xticks(range(len(MODELS)), [m[1] for m in MODELS])
    ax.set_ylim(0, 80)
    ax.set_ylabel('Replies that state the cover story (lie), %', fontsize=11, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='both', length=0, labelsize=10.5)
    ax.tick_params(axis='y', colors=MUTED)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9.5, handlelength=1.2)
    c = {k: s['models'][k]['contrasts'] for k, _ in MODELS}
    fig.text(0.04, 0.955, 'MASK honesty: models lie more in a deployment-looking rendering; scored examples that '
             'reward lying change nothing', fontsize=13.5, weight='bold', color=INK)
    fig.text(0.04, 0.92, '17 MASK scenarios (pressure to state a cover story in the system prompt), 6 renderings each, 5 samples '
             'per model. Lie/honest/evade judged by GPT-6-luna against MASK ground truth. Error bars: 95% Wilson.',
             fontsize=10.5, color=MUTED)
    lines = ['Paired, item-level (n=17): deployment-like minus as-shipped lying: ' + ', '.join(
        f'{name.replace(chr(10), " ")} {-100 * c[k]["E_minus_D"]["diff"]:+.0f} [{-100 * c[k]["E_minus_D"]["ci95"][1]:+.0f}, {-100 * c[k]["E_minus_D"]["ci95"][0]:+.0f}]'
        for k, name in MODELS) + '.',
        'Lying-rewarded minus honesty-rewarded examples (the costly reward test): ' + ', '.join(
        f'{name.replace(chr(10), " ")} {100 * c[k]["Slie_minus_Shonest"]["diff"]:+.0f} [{100 * c[k]["Slie_minus_Shonest"]["ci95"][0]:+.0f}, {100 * c[k]["Slie_minus_Shonest"]["ci95"][1]:+.0f}]'
        for k, name in MODELS) + '.',
        'Caveats: the deployment-like rewrite removes stakes language, which should reduce lying, so the increase runs against the confound;',
        'the detectability judge scores every rendering 0.97-1.00, so it cannot say whether the deployment-like version looked like deployment.']
    footer(fig, lines, 0.19)
    (OUT / 'analysis').mkdir(exist_ok=True)
    path = OUT / 'analysis' / 'mask-run.png'
    fig.savefig(path, facecolor='white')
    print('wrote', path)


if __name__ == '__main__':
    main()
