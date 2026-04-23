"""Generate publication-quality figures 1–5 for LLM Judge Consistency paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

CATEGORY_COLORS = {
    'writing':      '#e74c3c',
    'reasoning':    '#3498db',
    'coding':       '#2ecc71',
    'knowledge':    '#f39c12',
    'math':         '#9b59b6',
    'roleplay':     '#1abc9c',
    'extraction':   '#e67e22',
    'ethics':       '#34495e',
    'instruction':  '#e91e63',
    'hard':         '#607d8b',
}
JUDGE_COLORS = {'gpt-4o-mini': '#2980b9', 'gpt-4.1-mini': '#c0392b'}
JUDGE_LABELS = {'gpt-4o-mini': 'GPT-4o-mini', 'gpt-4.1-mini': 'GPT-4.1-mini'}

Path('paper/figures').mkdir(parents=True, exist_ok=True)

# ── Data loading ────────────────────────────────────────────────────────────
f = sorted(Path('results').glob('exp1_v2_*.json'))[-1]
raw = json.load(open(f))
# de-duplicate (judge, sample_id)
seen = {}
for item in raw:
    seen[(item['judge_model'], item['sample_id'])] = item
data = list(seen.values())


def flip_rate(item):
    pw = item.get('pairwise', [])
    if isinstance(pw, dict):
        pw = pw.get('judgments', [])
    winners = [j['winner'] for j in pw if isinstance(j, dict) and not j.get('error') and j.get('winner')]
    if not winners:
        return None
    c = Counter(winners)
    return 1.0 - c.most_common(1)[0][1] / len(winners)


def majority(item):
    pw = item.get('pairwise', [])
    if isinstance(pw, dict):
        pw = pw.get('judgments', [])
    winners = [j['winner'] for j in pw if isinstance(j, dict) and not j.get('error') and j.get('winner')]
    if not winners:
        return None
    return Counter(winners).most_common(1)[0][0]


# ── Figure 1: Flip rates per question, one panel per judge ─────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

for ax, judge in zip(axes, ['gpt-4o-mini', 'gpt-4.1-mini']):
    items = sorted([d for d in data if d['judge_model'] == judge],
                   key=lambda x: x['sample_id'])
    qids = [d['sample_id'] for d in items]
    flips = [flip_rate(d) * 100 for d in items]
    cats = [d.get('category', '') for d in items]
    bar_colors = [CATEGORY_COLORS.get(c, '#95a5a6') for c in cats]

    bars = ax.bar(range(len(qids)), flips, color=bar_colors, edgecolor='white',
                  linewidth=0.5, zorder=3)
    # value labels on bars > 5%
    for i, (b, v) in enumerate(zip(bars, flips)):
        if v > 5:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.0f}',
                    ha='center', va='bottom', fontsize=7, color='#333')

    mean_fr = np.mean(flips)
    ax.axhline(50, color='#e74c3c', linestyle='--', linewidth=1.2,
               alpha=0.7, label='Random baseline (50%)', zorder=2)
    ax.axhline(mean_fr, color='#2c3e50', linestyle=':', linewidth=1.2,
               alpha=0.8, label=f'Mean ({mean_fr:.1f}%)', zorder=2)
    ax.set_ylabel('Flip Rate (%)')
    ax.set_title(JUDGE_LABELS[judge], fontweight='bold')
    ax.set_ylim(0, 62)
    ax.set_xticks(range(len(qids)))
    ax.set_xticklabels(qids, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')

# Category legend
legend_handles = [mpatches.Patch(facecolor=CATEGORY_COLORS[c], label=c.capitalize())
                  for c in sorted(CATEGORY_COLORS)]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.03), title='Task Category', title_fontsize=9)

fig.suptitle('Pairwise Preference Flip Rates Across 29 Questions',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('paper/figures/fig1_flip_rates.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig1_flip_rates.png', bbox_inches='tight')
plt.close()
print("Fig 1 done")


# ── Figure 2: Position bias ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, judge in zip(axes, ['gpt-4o-mini', 'gpt-4.1-mini']):
    items = [d for d in data if d['judge_model'] == judge]
    maj = [majority(d) for d in items]
    c = Counter(m for m in maj if m)
    labels = ['A (first)', 'B (second)', 'Tie']
    vals = [c.get('A', 0), c.get('B', 0), c.get('tie', 0)]
    bar_colors = ['#e74c3c', '#3498db', '#95a5a6']
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, vals, color=bar_colors, edgecolor='white', width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.axhline(len(items) / 2, color='#2c3e50', linestyle=':', linewidth=1,
               alpha=0.6, label='Unbiased (50/50)')
    ax.set_title(JUDGE_LABELS[judge], fontweight='bold')
    ax.set_ylabel('Questions with majority preference')
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(vals) + 4)
    ax.legend(fontsize=8)

fig.suptitle('Position Bias: Which Response Wins the Majority?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig2_position_bias.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig2_position_bias.png', bbox_inches='tight')
plt.close()
print("Fig 2 done")


# ── Figure 3: Flip rate by category ────────────────────────────────────────
categories = sorted(set(d.get('category', '') for d in data if d.get('category')))
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(categories))
width = 0.35

for i, judge in enumerate(['gpt-4o-mini', 'gpt-4.1-mini']):
    means, sems = [], []
    for cat in categories:
        vals = [flip_rate(d) * 100 for d in data
                if d['judge_model'] == judge and d.get('category') == cat
                and flip_rate(d) is not None]
        means.append(np.mean(vals) if vals else 0)
        sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    bars = ax.bar(x + i * width, means, width, yerr=sems,
                  label=JUDGE_LABELS[judge], color=JUDGE_COLORS[judge],
                  alpha=0.85, capsize=4, error_kw={'linewidth': 1.2},
                  edgecolor='white', zorder=3)
    for b, v, s in zip(bars, means, sems):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + s + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Mean Flip Rate (%)')
ax.set_title('Intra-Judge Inconsistency by Task Category\n(error bars = ±1 SE)',
             fontweight='bold')
ax.set_xticks(x + width / 2)
ax.set_xticklabels([c.capitalize() for c in categories], rotation=30, ha='right')
ax.legend()
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
plt.tight_layout()
plt.savefig('paper/figures/fig3_category_flips.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig3_category_flips.png', bbox_inches='tight')
plt.close()
print("Fig 3 done")


# ── Figure 4: Pairwise flip rate vs pointwise score gap ────────────────────
fig, ax = plt.subplots(figsize=(7.5, 6))

for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    gaps, flips = [], []
    for item in [d for d in data if d['judge_model'] == judge]:
        fr = flip_rate(item)
        if fr is None:
            continue
        pa = [j['score'] for j in item.get('pointwise_a', [])
              if isinstance(j, dict) and not j.get('error') and j.get('score') is not None]
        pb = [j['score'] for j in item.get('pointwise_b', [])
              if isinstance(j, dict) and not j.get('error') and j.get('score') is not None]
        if pa and pb:
            gaps.append(abs(np.mean(pa) - np.mean(pb)))
            flips.append(fr * 100)

    ax.scatter(gaps, flips, marker='o' if judge == 'gpt-4o-mini' else 's',
               color=JUDGE_COLORS[judge], alpha=0.65, s=55,
               label=JUDGE_LABELS[judge], zorder=3)
    # trend line
    if len(gaps) > 2:
        z = np.polyfit(gaps, flips, 1)
        xline = np.linspace(min(gaps), max(gaps), 100)
        ax.plot(xline, np.poly1d(z)(xline), color=JUDGE_COLORS[judge],
                alpha=0.4, linewidth=1.5, linestyle='--')

ax.axhline(50, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5,
           label='Random baseline')
ax.set_xlabel('|Mean Score A − Mean Score B|  (on 1–10 scale)')
ax.set_ylabel('Pairwise Flip Rate (%)')
ax.set_title('Pairwise Inconsistency vs. Pointwise Score Gap\n'
             '(judges declare winners despite near-zero score differences)',
             fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('paper/figures/fig4_score_vs_flip.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig4_score_vs_flip.png', bbox_inches='tight')
plt.close()
print("Fig 4 done")


# ── Figure 5: Cross-judge flip rate heatmap ─────────────────────────────────
qids = sorted(set(d['sample_id'] for d in data))
cats_by_qid = {d['sample_id']: d.get('category', '') for d in data}
judges = ['gpt-4o-mini', 'gpt-4.1-mini']

matrix = []
for judge in judges:
    row = []
    for qid in qids:
        item = next((d for d in data if d['judge_model'] == judge and d['sample_id'] == qid), None)
        fr = flip_rate(item) * 100 if item and flip_rate(item) is not None else 0
        row.append(fr)
    matrix.append(row)

fig, ax = plt.subplots(figsize=(14, 3.5))
im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=50)
plt.colorbar(im, ax=ax, label='Flip Rate (%)', shrink=0.8)

ax.set_yticks([0, 1])
ax.set_yticklabels([JUDGE_LABELS[j] for j in judges])
ax.set_xticks(range(len(qids)))
ax.set_xticklabels(qids, rotation=45, ha='right', fontsize=7)

# annotate cells
for i in range(len(judges)):
    for j in range(len(qids)):
        v = matrix[i][j]
        text_color = 'white' if v > 30 else '#333'
        ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                fontsize=6.5, color=text_color, fontweight='bold')

# add category color bar at top
cat_ax = ax.twiny()
cat_ax.set_xlim(ax.get_xlim())
cat_ax.set_xticks(range(len(qids)))
cat_ax.set_xticklabels([cats_by_qid.get(q, '')[:3] for q in qids],
                        rotation=45, ha='left', fontsize=7,
                        color='#555')
cat_ax.tick_params(length=0)

ax.set_title('Cross-Judge Flip Rate Heatmap  (darker = more inconsistent)',
             fontweight='bold', pad=28)
plt.tight_layout()
plt.savefig('paper/figures/fig5_heatmap.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig5_heatmap.png', bbox_inches='tight')
plt.close()
print("Fig 5 done")

print("\nFigures 1–5 saved to paper/figures/")
