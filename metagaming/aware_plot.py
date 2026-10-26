"""Figures for the EvalDetectBench-based run. Offline; reads analysis/summary.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .aware_bench import OUT, DEPTHS
from .habit_plot import INK, MUTED, GRID, footer

MODELS = [('luna', 'GPT-6-luna', '#2A78D6'), ('deepseek', 'DeepSeek V4.1 Flash', '#1BAF7A'),
          ('gptoss-high', 'gpt-oss-120b high effort', '#EB6834'), ('gptoss-low', 'gpt-oss-120b low effort', '#B9B7AF')]


def style(ax, ylabel, ymax=100):
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=10.5, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='both', length=0, labelsize=10)
    ax.tick_params(axis='y', colors=MUTED)


def main():
    s = json.loads((OUT / 'analysis' / 'summary.json').read_text())
    plt.rcParams.update({'font.family': 'Arial', 'text.color': INK, 'xtick.color': INK})
    fig = plt.figure(figsize=(12.5, 8.2), dpi=220)
    ax1 = fig.add_axes([0.07, 0.36, 0.26, 0.40])
    ax2 = fig.add_axes([0.40, 0.36, 0.30, 0.40])
    ax3 = fig.add_axes([0.77, 0.36, 0.20, 0.40])
    # panel 1: detectability by depth
    pe = [100 * s['detect'][f'chat_depth{d}_mean'] for d in DEPTHS]
    ax1.plot(range(4), pe, color=INK, marker='o', linewidth=2.2, zorder=3, label="A harmful request placed after a real user's chat history")
    dep = 100 * s['detect']['deployment']['mean_p_eval']
    ax1.axhline(dep, color='#8A8880', linestyle=(0, (3, 3)), linewidth=1.5, label='The same real chat histories with no request added')
    for x, y in zip(range(4), pe):
        ax1.text(x, y + 3, f'{y:.0f}%', ha='center', fontsize=9.5, color=INK)
    ax1.text(3.05, dep - 6, f'{dep:.0f}%', fontsize=9.5, color='#8A8880', ha='right')
    ax1.set_xticks(range(4), [f'depth {d}' for d in DEPTHS])
    style(ax1, 'Judge P(eval) (% "evaluation" votes)')
    ax1.set_title('Detectability does not fall with depth', fontsize=11, loc='left', color=INK, pad=40)
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.5, handlelength=1.5)
    # panel 2: compliance by depth per model
    ends = []
    for key, name, color in MODELS:
        ys = [100 * s['behave'][key]['chat'][f'depth{d}']['comply'] for d in DEPTHS]
        ax2.plot(range(4), ys, color=color, marker='o', linewidth=2.2, zorder=3, label=name)
        ends.append((ys[0], ys[-1], color))
    for y0, y1, color in ends:
        ax2.text(-0.12, y0, f'{y0:.0f}%', fontsize=8.5, color=color, va='center', ha='right')
        ax2.text(3.12, y1, f'{y1:.0f}%', fontsize=8.5, color=color, va='center')
    ax2.set_xticks(range(4), [f'depth {d}' for d in DEPTHS])
    ax2.set_xlim(-0.3, 3.5)
    style(ax2, 'Compliance with the request (%)')
    ax2.set_title('Behavior barely moves with depth', fontsize=11, loc='left', color=INK, pad=40)
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, handlelength=1.5)
    # panel 3: agentic harmful rate with / without notice
    w = 0.36
    for i, (key, name, color) in enumerate(MODELS):
        for j, cue in enumerate(('notice', 'no_notice')):
            v = s['behave'][key]['agentic'][cue]
            x = i + (j - 0.5) * w
            ax3.bar(x, 100 * v['harmful'], w, color=color if j == 0 else 'white', edgecolor=color, linewidth=1.5, zorder=3)
            ax3.text(x, 100 * v['harmful'] + 1, f'{100 * v["harmful"]:.0f}', ha='center', fontsize=8.5, color=INK)
    ax3.set_xticks(range(4), ['luna', 'deepseek', 'oss-high', 'oss-low'], fontsize=9)
    style(ax3, 'Harmful action (blackmail or leak), %', 40)
    ax3.set_title('Agentic: eval notice changes nothing', fontsize=11, loc='left', color=INK, pad=40)
    ax3.text(3.5, 36, 'filled = with notice\nopen = notice removed', fontsize=8.5, color=MUTED, ha='right', va='top')
    fig.text(0.04, 0.955, 'Putting a harmful request after a real chat history makes it look MORE like a test, not less; '
             'behavior does not change', fontsize=13, weight='bold', color=INK)
    fig.text(0.04, 0.92, '40 safety requests (StrongREJECT, Do-Not-Answer, XSTest) at 4 depths, 3 samples per model; '
             '20 Agentic Misalignment scenarios with/without the shipped notice, 3 samples.',
             fontsize=10.5, color=MUTED)
    footer(fig, ['Depth d = the request arrives after d messages of a real deployment chat from the dataset. P(eval): GPT-6-luna '
                 'judge, 4 EvalDetectBench probe questions, mean "evaluation" vote; the dashed line is the same judge on the',
                 'replayed deployment chats without any appended request. Compliance judged by GPT-6-luna. Paired depth-8 minus '
                 'depth-0 compliance: luna +1 pt, DeepSeek +6 [-3, +16], gpt-oss high +1, gpt-oss low -7 [-16, -1].',
                 'Reading: the harmful request itself is the evaluation cue; wrapping it in real conversation does not hide it, '
                 'so this run did not produce a deployment-like rendering and therefore could not test for a gap.'], 0.2)
    (OUT / 'analysis').mkdir(exist_ok=True)
    path = OUT / 'analysis' / 'first-run.png'
    fig.savefig(path, facecolor='white')
    print('wrote', path)


if __name__ == '__main__':
    main()
