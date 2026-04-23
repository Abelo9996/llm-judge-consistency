"""Generate Figure 6: Temperature ablation — connected scatter plot."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'figure.dpi': 150,
})

JUDGE_COLORS = {'gpt-4o-mini': '#2980b9', 'gpt-4.1-mini': '#c0392b'}
JUDGE_LABELS = {'gpt-4o-mini': 'GPT-4o-mini', 'gpt-4.1-mini': 'GPT-4.1-mini'}


def get_flip_rates(filepath, judge):
    d = json.load(open(filepath))
    result = {}
    for item in d:
        if item['judge_model'] != judge:
            continue
        pw = item.get('pairwise', [])
        if isinstance(pw, dict):
            pw = pw.get('judgments', [])
        winners = [j['winner'] for j in pw
                   if isinstance(j, dict) and not j.get('error') and j.get('winner')]
        if winners:
            c = Counter(winners)
            result[item['sample_id']] = 1 - c.most_common(1)[0][1] / len(winners)
    return result


f1 = sorted(Path('results').glob('exp1_v2_*.json'))[-1]
f0 = sorted(Path('results').glob('exp2_temp0_*.json'))[-1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, judge in zip(axes, ['gpt-4o-mini', 'gpt-4.1-mini']):
    t1 = get_flip_rates(f1, judge)
    t0 = get_flip_rates(f0, judge)
    color = JUDGE_COLORS[judge]

    qids = sorted(set(t1.keys()) & set(t0.keys()))
    vals_t1 = np.array([t1[q] * 100 for q in qids])
    vals_t0 = np.array([t0[q] * 100 for q in qids])

    # draw connecting lines first
    for v1, v0 in zip(vals_t1, vals_t0):
        ax.plot([0, 1], [v1, v0], color=color, alpha=0.25, linewidth=1)

    # scatter points
    jitter = np.random.RandomState(42).uniform(-0.04, 0.04, len(qids))
    ax.scatter(np.zeros(len(qids)) + jitter, vals_t1, color=color,
               alpha=0.75, s=45, zorder=4, label=f't=1.0 (N=50 trials)')
    ax.scatter(np.ones(len(qids)) + jitter, vals_t0, color=color,
               marker='s', alpha=0.75, s=45, zorder=4,
               label=f't=0 (N=10 trials, approx.)')

    # mean lines
    mean_t1, mean_t0 = np.mean(vals_t1), np.mean(vals_t0)
    reduction = (1 - mean_t0 / mean_t1) * 100 if mean_t1 > 0 else 0
    ax.hlines(mean_t1, -0.25, 0.25, colors=color, linewidths=2, linestyles='-', zorder=5)
    ax.hlines(mean_t0, 0.75, 1.25, colors=color, linewidths=2, linestyles='-', zorder=5)
    ax.annotate(f'mean={mean_t1:.1f}%', xy=(0, mean_t1), xytext=(-0.4, mean_t1),
                fontsize=9, va='center', color=color, fontweight='bold')
    ax.annotate(f'mean={mean_t0:.1f}%\n(↓{reduction:.0f}%)', xy=(1, mean_t0),
                xytext=(1.05, mean_t0), fontsize=9, va='center', color=color, fontweight='bold')

    ax.set_xlim(-0.5, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Temperature = 1.0\n(default)', 'Temperature = 0\n(deterministic)'],
                        fontsize=10)
    ax.set_ylabel('Flip Rate (%)')
    ax.set_ylim(-2, 62)
    ax.set_title(JUDGE_LABELS[judge], fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(50, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(1.55, 50.5, 'Random', fontsize=8, color='#e74c3c', alpha=0.7)

fig.suptitle('Temperature Ablation: Flip Rates at t=1.0 vs t=0\n'
             '(each line = one question; t=0 N=10 is approximate vs t=1.0 N=50)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig6_temp_ablation.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig6_temp_ablation.png', bbox_inches='tight')
plt.close()
print("Fig 6 done")

for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    t1 = get_flip_rates(f1, judge)
    t0 = get_flip_rates(f0, judge)
    m1 = np.mean(list(t1.values())) * 100
    m0 = np.mean(list(t0.values())) * 100
    print(f"{JUDGE_LABELS[judge]}: t=1 mean={m1:.1f}%  t=0 mean={m0:.1f}%  reduction={100*(1-m0/m1):.0f}%")
