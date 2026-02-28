"""Generate all figures for Paper 2: LLM Judge Consistency."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Setup
Path('paper/figures').mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 300})

# Load data
f = sorted(Path('results').glob('exp1_v2_*.json'))[-1]
data = json.load(open(f))
unique = {}
for item in data:
    unique[(item['judge_model'], item['sample_id'])] = item
data = list(unique.values())

def get_flip_rate(item):
    pw = item.get('pairwise', [])
    if isinstance(pw, dict): pw = pw.get('judgments', [])
    winners = [j.get('winner','?') for j in pw if isinstance(j,dict) and not j.get('error') and 'winner' in j]
    if not winners: return None
    c = Counter(winners)
    return 1 - c.most_common(1)[0][1] / len(winners)

def get_majority(item):
    pw = item.get('pairwise', [])
    if isinstance(pw, dict): pw = pw.get('judgments', [])
    winners = [j.get('winner','?') for j in pw if isinstance(j,dict) and not j.get('error') and 'winner' in j]
    if not winners: return None
    c = Counter(winners)
    return c.most_common(1)[0][0]

# ============================================================
# Figure 1: Flip rates by question, grouped by judge
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for idx, judge in enumerate(['gpt-4o-mini', 'gpt-4.1-mini']):
    items = sorted([d for d in data if d['judge_model']==judge], key=lambda x: x['sample_id'])
    qids = [d['sample_id'] for d in items]
    flips = [get_flip_rate(d)*100 for d in items]
    cats = [d.get('category','') for d in items]
    
    colors = {'writing':'#e74c3c','reasoning':'#3498db','coding':'#2ecc71','knowledge':'#f39c12',
              'math':'#9b59b6','roleplay':'#1abc9c','extraction':'#e67e22','ethics':'#34495e',
              'instruction':'#e91e63','hard':'#607d8b'}
    
    bar_colors = [colors.get(c, '#95a5a6') for c in cats]
    axes[idx].bar(range(len(qids)), flips, color=bar_colors, alpha=0.85, edgecolor='white')
    axes[idx].set_ylabel('Flip Rate (%)')
    axes[idx].set_title(f'{judge}', fontweight='bold')
    axes[idx].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    axes[idx].axhline(y=np.mean(flips), color='black', linestyle=':', alpha=0.7, label=f'Mean ({np.mean(flips):.1f}%)')
    axes[idx].legend(fontsize=9)
    axes[idx].set_ylim(0, 60)

axes[1].set_xticks(range(len(qids)))
axes[1].set_xticklabels(qids, rotation=45, ha='right', fontsize=8)

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[c], label=c) for c in sorted(colors)]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Position Bias: Pairwise Preference Flip Rates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig1_flip_rates.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig1_flip_rates.png', bbox_inches='tight')
plt.close()
print("Fig 1 done")

# ============================================================
# Figure 2: Position bias — A vs B majority wins
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for idx, judge in enumerate(['gpt-4o-mini', 'gpt-4.1-mini']):
    items = [d for d in data if d['judge_model']==judge]
    majorities = [get_majority(d) for d in items]
    c = Counter(majorities)
    labels = ['A (first)', 'B (second)', 'tie']
    vals = [c.get('A',0), c.get('B',0), c.get('tie',0)]
    colors_pie = ['#e74c3c', '#3498db', '#95a5a6']
    axes[idx].bar(labels, vals, color=colors_pie, edgecolor='white')
    axes[idx].set_title(f'{judge}', fontweight='bold')
    axes[idx].set_ylabel('Questions won (majority)')
    for i, v in enumerate(vals):
        axes[idx].text(i, v+0.3, str(v), ha='center', fontweight='bold')
    axes[idx].set_ylim(0, 30)

plt.suptitle('Position Bias: Which Response Wins Majority?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig2_position_bias.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig2_position_bias.png', bbox_inches='tight')
plt.close()
print("Fig 2 done")

# ============================================================
# Figure 3: Flip rate by category (both judges)
# ============================================================
categories = sorted(set(d.get('category','') for d in data))
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(categories))
width = 0.35

for idx, judge in enumerate(['gpt-4o-mini', 'gpt-4.1-mini']):
    means = []
    for cat in categories:
        items = [d for d in data if d['judge_model']==judge and d.get('category','')==cat]
        flips = [get_flip_rate(d)*100 for d in items if get_flip_rate(d) is not None]
        means.append(np.mean(flips) if flips else 0)
    ax.bar(x + idx*width, means, width, label=judge, alpha=0.85)

ax.set_ylabel('Mean Flip Rate (%)')
ax.set_title('Flip Rate by Task Category', fontsize=13, fontweight='bold')
ax.set_xticks(x + width/2)
ax.set_xticklabels(categories, rotation=30, ha='right')
ax.legend()
ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('paper/figures/fig3_category_flips.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig3_category_flips.png', bbox_inches='tight')
plt.close()
print("Fig 3 done")

# ============================================================
# Figure 4: Pointwise scores vs pairwise disagreement
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
for judge, marker, color in [('gpt-4o-mini','o','#3498db'), ('gpt-4.1-mini','s','#e74c3c')]:
    items = [d for d in data if d['judge_model']==judge]
    for item in items:
        flip = get_flip_rate(item)
        if flip is None: continue
        pa = item.get('pointwise_a', [])
        pb = item.get('pointwise_b', [])
        a_scores = [j['score'] for j in pa if isinstance(j,dict) and not j.get('error') and j.get('score') is not None]
        b_scores = [j['score'] for j in pb if isinstance(j,dict) and not j.get('error') and j.get('score') is not None]
        if a_scores and b_scores:
            gap = abs(np.mean(a_scores) - np.mean(b_scores))
            ax.scatter(gap, flip*100, marker=marker, color=color, alpha=0.6, s=60)

ax.scatter([], [], marker='o', color='#3498db', label='gpt-4o-mini')
ax.scatter([], [], marker='s', color='#e74c3c', label='gpt-4.1-mini')
ax.set_xlabel('|Score_A - Score_B| (Pointwise Gap)')
ax.set_ylabel('Flip Rate (%)')
ax.set_title('Pairwise Inconsistency vs Pointwise Score Difference', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('paper/figures/fig4_score_vs_flip.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig4_score_vs_flip.png', bbox_inches='tight')
plt.close()
print("Fig 4 done")

# ============================================================
# Figure 5: Cross-judge agreement heatmap
# ============================================================
qids = sorted(set(d['sample_id'] for d in data))
judges = ['gpt-4o-mini', 'gpt-4.1-mini']

fig, ax = plt.subplots(figsize=(12, 4))
matrix = []
for judge in judges:
    row = []
    for qid in qids:
        item = next((d for d in data if d['judge_model']==judge and d['sample_id']==qid), None)
        if item:
            flip = get_flip_rate(item)
            row.append(flip*100 if flip is not None else 0)
        else:
            row.append(0)
    matrix.append(row)

im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=50)
ax.set_yticks([0,1])
ax.set_yticklabels(judges)
ax.set_xticks(range(len(qids)))
ax.set_xticklabels(qids, rotation=45, ha='right', fontsize=7)
plt.colorbar(im, label='Flip Rate (%)')
ax.set_title('Flip Rate Heatmap Across Questions and Judges', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig5_heatmap.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig5_heatmap.png', bbox_inches='tight')
plt.close()
print("Fig 5 done")

print("\nAll figures saved to paper/figures/")
