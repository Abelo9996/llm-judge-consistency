"""Generate Figure 8: Reliability curve — P(K-trial majority matches 50-trial majority)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

metrics = json.load(open('results/metrics_advanced.json'))
rc = metrics['reliability_curve']

Kvals       = rc['k_values']
mean_c      = np.array(rc['mean_correct'])
se_c        = np.array(rc['se_correct'])
hard_c      = np.array(rc['hard_questions'])
easy_c      = np.array(rc['easy_questions'])
by_judge    = rc['by_judge']
min_n_95    = rc['overall_min_n_95']
min_n_90    = rc['overall_min_n_90']
hard_n_90   = rc['hard_min_n_90']

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Left panel: overall + strata ────────────────────────────────────────────
ax = axes[0]
ax.fill_between(Kvals, mean_c - se_c, mean_c + se_c,
                color='#2c3e50', alpha=0.15, label='_nolegend_')
ax.plot(Kvals, mean_c, color='#2c3e50', linewidth=2.2, label='All questions (mean ± SE)')
ax.plot(Kvals, easy_c, color='#27ae60', linewidth=1.8, linestyle='--',
        label=f'Easy questions (FR < 10%, n={sum(1 for v in metrics["per_item_min_n"].values() if v["flip_rate"] < 0.10)})')
ax.plot(Kvals, hard_c, color='#e74c3c', linewidth=1.8, linestyle='--',
        label=f'Hard questions (FR ≥ 10%, n={sum(1 for v in metrics["per_item_min_n"].values() if v["flip_rate"] >= 0.10)})')

# Reference lines
ax.axhline(0.95, color='#8e44ad', linestyle=':', linewidth=1.3, alpha=0.8)
ax.axhline(0.90, color='#d35400', linestyle=':', linewidth=1.3, alpha=0.8)
ax.text(51.5, 0.951, '95%', va='center', fontsize=9, color='#8e44ad')
ax.text(51.5, 0.901, '90%', va='center', fontsize=9, color='#d35400')

# Annotate key thresholds
ax.axvline(min_n_95, color='#8e44ad', linestyle=':', linewidth=1, alpha=0.6)
ax.axvline(hard_n_90, color='#e74c3c', linestyle=':', linewidth=1, alpha=0.5)
ax.annotate(f'K={min_n_95}\n(overall 95%)',
            xy=(min_n_95, 0.95), xytext=(min_n_95 + 3, 0.88),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.2),
            fontsize=8.5, color='#8e44ad')
ax.annotate(f'K={hard_n_90}\n(hard q. 90%)',
            xy=(hard_n_90, hard_c[hard_n_90 - 1]), xytext=(hard_n_90 + 4, 0.82),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
            fontsize=8.5, color='#e74c3c')

ax.set_xlabel('Number of Trials (K)')
ax.set_ylabel('P(K-trial majority = 50-trial majority)')
ax.set_title('Reliability Curve: Easy vs Hard Questions', fontweight='bold')
ax.set_xlim(1, 55)
ax.set_ylim(0.60, 1.02)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

# ── Right panel: per-judge curves + scatter of min-N ────────────────────────
ax2 = axes[1]
JUDGE_COLORS = {'gpt-4o-mini': '#2980b9', 'gpt-4.1-mini': '#c0392b'}
JUDGE_LABELS = {'gpt-4o-mini': 'GPT-4o-mini', 'gpt-4.1-mini': 'GPT-4.1-mini'}
for judge, curve in by_judge.items():
    ax2.plot(Kvals, curve, color=JUDGE_COLORS[judge], linewidth=2,
             label=JUDGE_LABELS[judge])

ax2.axhline(0.95, color='#8e44ad', linestyle=':', linewidth=1.3, alpha=0.8)
ax2.axhline(0.90, color='#d35400', linestyle=':', linewidth=1.3, alpha=0.8)
ax2.text(51.5, 0.951, '95%', va='center', fontsize=9, color='#8e44ad')
ax2.text(51.5, 0.901, '90%', va='center', fontsize=9, color='#d35400')

# Scatter: per-item min-N vs flip rate
ax2b = ax2.twinx()
for item_key, item_data in metrics['per_item_min_n'].items():
    judge = item_data['judge']
    color = JUDGE_COLORS.get(judge, '#95a5a6')
    ax2b.scatter(item_data['flip_rate'] * 100, item_data['min_n_90'],
                 color=color, alpha=0.4, s=28, zorder=2)

ax2b.set_ylabel('Min trials for 90% accuracy (per question, right axis)',
                color='#555', fontsize=9)
ax2b.tick_params(axis='y', labelcolor='#555', labelsize=8)
ax2b.set_ylim(0, 52)

ax2.set_xlabel('Number of Trials (K)')
ax2.set_ylabel('P(K-trial majority = 50-trial majority)')
ax2.set_title('Reliability by Judge + Min-N vs Flip Rate', fontweight='bold')
ax2.set_xlim(1, 55)
ax2.set_ylim(0.60, 1.02)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')

fig.suptitle(
    'Figure 8: How Many Trials Are Needed for a Stable Majority Verdict?\n'
    '(Monte Carlo subsampling, 500 repetitions per question per K)',
    fontsize=11, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('paper/figures/fig8_reliability_curve.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig8_reliability_curve.png', bbox_inches='tight')
plt.close()
print("Fig 8 done")
print(f"  Overall 95% threshold at K={min_n_95}")
print(f"  Overall 90% threshold at K={min_n_90}")
print(f"  Hard questions 90% threshold at K={hard_n_90}")
