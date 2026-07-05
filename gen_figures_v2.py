import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path('/Users/abelyagubyan/Downloads/llm-judge-consistency')
FIGDIR = ROOT / 'paper' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

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
    'grid.alpha': 0.18,
    'grid.linestyle': '--',
})

CATEGORY_COLORS = {
    'writing': '#d95f02',
    'reasoning': '#1b9e77',
    'coding': '#7570b3',
    'knowledge': '#66a61e',
    'math': '#e7298a',
    'roleplay': '#a6761d',
    'extraction': '#e6ab02',
    'ethics': '#666666',
    'instruction': '#1f78b4',
    'hard': '#b15928',
}
JUDGE_COLORS = {'gpt-4o-mini': '#4c78a8', 'gpt-4.1-mini': '#e45756'}
JUDGE_LABELS = {'gpt-4o-mini': 'GPT-4o-mini', 'gpt-4.1-mini': 'GPT-4.1-mini'}

raw = json.load(open(sorted((ROOT / 'results').glob('exp1_v2_*.json'))[-1]))
seen = {}
for item in raw:
    seen[(item['judge_model'], item['sample_id'])] = item
DATA = list(seen.values())
ADV = json.load(open(ROOT / 'results' / 'metrics_advanced.json'))


def get_pairwise(item):
    pw = item.get('pairwise', [])
    if isinstance(pw, dict):
        pw = pw.get('judgments', [])
    return [j for j in pw if isinstance(j, dict) and not j.get('error') and j.get('winner')]


def flip_rate(item):
    winners = [j['winner'] for j in get_pairwise(item)]
    c = Counter(winners)
    return 1 - c.most_common(1)[0][1] / len(winners)


def majority(item):
    winners = [j['winner'] for j in get_pairwise(item)]
    return Counter(winners).most_common(1)[0][0]


# Figure 1: sorted horizontal paired bars
rows = []
for qid in sorted({d['sample_id'] for d in DATA}):
    per = {d['judge_model']: d for d in DATA if d['sample_id'] == qid}
    fr1 = flip_rate(per['gpt-4o-mini']) * 100
    fr2 = flip_rate(per['gpt-4.1-mini']) * 100
    cat = per['gpt-4o-mini'].get('category', '')
    rows.append((qid, cat, fr1, fr2, (fr1 + fr2) / 2))
rows.sort(key=lambda x: x[4])
labels = [f"{qid} ({cat})" for qid, cat, *_ in rows]
y = np.arange(len(rows))
fr1 = [r[2] for r in rows]
fr2 = [r[3] for r in rows]
cat_colors = [CATEGORY_COLORS[r[1]] for r in rows]

fig, ax = plt.subplots(figsize=(10, 11))
ax.barh(y - 0.18, fr1, height=0.34, color=JUDGE_COLORS['gpt-4o-mini'], label='GPT-4o-mini', alpha=0.92)
ax.barh(y + 0.18, fr2, height=0.34, color=JUDGE_COLORS['gpt-4.1-mini'], label='GPT-4.1-mini', alpha=0.88)
ax.axvline(13.6, color='#444', linestyle=':', linewidth=1.2, label='Overall mean (13.6%)')
ax.axvspan(40, 50, color='#fdd0a2', alpha=0.35, label='Coin-flip danger zone')
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel('Flip Rate (%)')
ax.set_title('Pairwise flip rates by question, sorted by mean instability', fontweight='bold')
ax.set_xlim(0, 60)
for idx in [0, 1, 2, len(rows)-3, len(rows)-2, len(rows)-1]:
    ax.text(max(fr1[idx], fr2[idx]) + 0.8, y[idx], f"{rows[idx][4]:.0f}%", va='center', fontsize=8)
legend_handles = [mpatches.Patch(color=c, label=k.capitalize()) for k, c in CATEGORY_COLORS.items()]
leg1 = ax.legend(loc='lower right')
ax.add_artist(leg1)
ax.legend(handles=legend_handles, loc='upper right', title='Category', ncol=1, frameon=True)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig1_flip_rates.pdf', bbox_inches='tight')
plt.savefig(FIGDIR / 'fig1_flip_rates.png', bbox_inches='tight')
plt.close()

# Figure 3: paradox scatter improved
fig, ax = plt.subplots(figsize=(7.8, 6.2))
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    gaps, flips, qids = [], [], []
    for item in [d for d in DATA if d['judge_model'] == judge]:
        pa = [j['score'] for j in item.get('pointwise_a', []) if isinstance(j, dict) and not j.get('error') and j.get('score') is not None]
        pb = [j['score'] for j in item.get('pointwise_b', []) if isinstance(j, dict) and not j.get('error') and j.get('score') is not None]
        if pa and pb:
            gaps.append(abs(np.mean(pa) - np.mean(pb)))
            flips.append(flip_rate(item) * 100)
            qids.append(item['sample_id'])
    ax.scatter(gaps, flips, s=62, alpha=0.72, color=JUDGE_COLORS[judge], label=JUDGE_LABELS[judge], edgecolor='white', linewidth=0.5)
    z = np.polyfit(gaps, flips, 1)
    xs = np.linspace(min(gaps), max(gaps), 100)
    ax.plot(xs, np.poly1d(z)(xs), linestyle='--', color=JUDGE_COLORS[judge], alpha=0.7)
    for g, f, q in sorted(zip(gaps, flips, qids), key=lambda t: (-t[1], t[0]))[:2]:
        ax.annotate(q, (g, f), textcoords='offset points', xytext=(4, 4), fontsize=8)
ax.axvspan(0, 0.5, color='#d9f0d3', alpha=0.3)
ax.axhspan(20, 60, color='#fee0d2', alpha=0.22)
ax.text(0.07, 52, 'Paradox zone', fontsize=9, color='#7f0000')
ax.set_xlabel('Absolute mean pointwise score gap')
ax.set_ylabel('Pairwise flip rate (%)')
ax.set_title('Pairwise--pointwise gap: unstable verdicts despite tiny score differences', fontweight='bold')
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 60)
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / 'fig4_score_vs_flip.pdf', bbox_inches='tight')
plt.savefig(FIGDIR / 'fig4_score_vs_flip.png', bbox_inches='tight')
plt.close()

# Figure 4 category dot plot
cats = sorted({d.get('category','') for d in DATA}, key=lambda c: np.mean([flip_rate(d)*100 for d in DATA if d.get('category')==c]), reverse=True)
fig, ax = plt.subplots(figsize=(8.8, 5.8))
for i, judge in enumerate(['gpt-4o-mini', 'gpt-4.1-mini']):
    means = []
    ses = []
    ns = []
    for c in cats:
        vals = [flip_rate(d)*100 for d in DATA if d['judge_model']==judge and d.get('category')==c]
        means.append(np.mean(vals))
        ses.append(np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0)
        ns.append(len(vals))
    x = np.array(means)
    yv = np.arange(len(cats)) + (-0.12 if i==0 else 0.12)
    ax.errorbar(x, yv, xerr=ses, fmt='o', color=JUDGE_COLORS[judge], capsize=3, label=JUDGE_LABELS[judge])
ax.axvline(13.6, linestyle=':', color='#444', linewidth=1.2, label='Overall mean')
ax.set_yticks(np.arange(len(cats)))
ax.set_yticklabels([f"{c.capitalize()} (n={len([d for d in DATA if d.get('category')==c and d['judge_model']=='gpt-4o-mini'])})" for c in cats])
ax.invert_yaxis()
ax.set_xlabel('Mean flip rate (%)')
ax.set_title('Flip rate varies substantially by task category', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / 'fig3_category_flips.pdf', bbox_inches='tight')
plt.savefig(FIGDIR / 'fig3_category_flips.png', bbox_inches='tight')
plt.close()

# Figure 5 heatmap judges x categories
cats = sorted({d.get('category','') for d in DATA}, key=lambda c: np.mean([flip_rate(d)*100 for d in DATA if d.get('category')==c]), reverse=True)
mat = np.array([[np.mean([flip_rate(d)*100 for d in DATA if d['judge_model']==j and d.get('category')==c]) for c in cats] for j in ['gpt-4o-mini','gpt-4.1-mini']])
fig, ax = plt.subplots(figsize=(10.5, 3.3))
im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=40)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.1f}", ha='center', va='center', fontsize=9, color='black')
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([c.capitalize() for c in cats], rotation=25, ha='right')
ax.set_yticks([0,1])
ax.set_yticklabels(['GPT-4o-mini','GPT-4.1-mini'])
ax.set_title('Judge × category heatmap of mean flip rates', fontweight='bold')
cb = plt.colorbar(im, ax=ax, shrink=0.85)
cb.set_label('Mean flip rate (%)')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig5_heatmap.pdf', bbox_inches='tight')
plt.savefig(FIGDIR / 'fig5_heatmap.png', bbox_inches='tight')
plt.close()

# Figure 6 temperature slopegraph
exp2 = json.load(open(sorted((ROOT / 'results').glob('exp2_temp0_*.json'))[-1]))
# infer per judge/question temp0 FR
exp2map = {}
for item in exp2:
    key = (item['judge_model'], item['sample_id'])
    exp2map[key] = item
fig, axes = plt.subplots(1,2, figsize=(10.5,6), sharey=True)
for ax, judge in zip(axes, ['gpt-4o-mini','gpt-4.1-mini']):
    vals = []
    for qid in sorted({d['sample_id'] for d in DATA if d['judge_model']==judge}):
        t1 = flip_rate(next(d for d in DATA if d['judge_model']==judge and d['sample_id']==qid))*100
        item = exp2map[(judge,qid)]
        pw = item['pairwise']['judgments'] if isinstance(item.get('pairwise'),dict) else item.get('pairwise',[])
        wins = [j['winner'] for j in pw if isinstance(j,dict) and not j.get('error') and j.get('winner')]
        fr0 = (1 - Counter(wins).most_common(1)[0][1]/len(wins))*100
        vals.append((qid,t1,fr0))
    for qid,t1,fr0 in vals:
        color = '#2ca25f' if fr0 < t1 else '#de2d26'
        ax.plot([0,1],[t1,fr0], color=color, alpha=0.6, linewidth=1.5)
        if fr0 >= 45 or t1 >= 45:
            ax.annotate(qid, (1,fr0), textcoords='offset points', xytext=(4,0), fontsize=8)
    ax.scatter([0]*len(vals), [v[1] for v in vals], color=JUDGE_COLORS[judge], s=28, zorder=3)
    ax.scatter([1]*len(vals), [v[2] for v in vals], color=JUDGE_COLORS[judge], s=28, zorder=3)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['t=1.0','t=0'])
    ax.set_title(JUDGE_LABELS[judge], fontweight='bold')
    ax.set_ylim(0,60)
axes[0].set_ylabel('Flip rate (%)')
fig.suptitle('Deterministic decoding reduces, but does not eliminate, inconsistency', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig6_temp_ablation.pdf', bbox_inches='tight')
plt.savefig(FIGDIR / 'fig6_temp_ablation.png', bbox_inches='tight')
plt.close()

# Figure 7 prompt sensitivity slopegraph
exp3 = json.load(open(sorted((ROOT / 'results').glob('exp3_prompt_*.json'))[-1]))
pmap = defaultdict(dict)
for item in exp3:
    pmap[(item['judge_model'], item['sample_id'])][item['prompt_template']] = item
fig, axes = plt.subplots(1,2, figsize=(10.5,6), sharey=True)
for ax, judge in zip(axes, ['gpt-4o-mini','gpt-4.1-mini']):
    keys = sorted([k for k in pmap if k[0]==judge])
    for _, qid in keys:
        d = pmap[(judge,qid)]
        if 'prompt_A' not in d or 'prompt_B' not in d:
            continue
        def fr(item):
            pw = item['pairwise']['judgments'] if isinstance(item.get('pairwise'),dict) else item.get('pairwise',[])
            wins = [j['winner'] for j in pw if isinstance(j,dict) and not j.get('error') and j.get('winner')]
            return (1 - Counter(wins).most_common(1)[0][1]/len(wins))*100
        a, b = fr(d['prompt_A']), fr(d['prompt_B'])
        maj_a = majority(d['prompt_A']); maj_b = majority(d['prompt_B'])
        color = '#cb181d' if maj_a != maj_b else '#6b6b6b'
        ax.plot([0,1],[a,b], color=color, alpha=0.75, linewidth=1.9)
        ax.scatter([0,1], [a,b], color=color, s=28, zorder=3)
        if maj_a != maj_b:
            ax.annotate(qid, (1,b), textcoords='offset points', xytext=(4,0), fontsize=8)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Prompt A','Prompt B'])
    ax.set_title(JUDGE_LABELS[judge], fontweight='bold')
    ax.set_ylim(0,60)
axes[0].set_ylabel('Flip rate (%)')
fig.suptitle('Prompt wording changes both instability and majority verdicts', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig7_prompt_sensitivity.pdf', bbox_inches='tight')
plt.savefig(FIGDIR / 'fig7_prompt_sensitivity.png', bbox_inches='tight')
plt.close()

# Figure 8 reliability curve cleaner
k = ADV['reliability_curve']['k_values']
mean = np.array(ADV['reliability_curve']['mean_correct']) * 100
hard = np.array(ADV['reliability_curve']['hard_questions']) * 100
easy = np.array(ADV['reliability_curve']['easy_questions']) * 100
fig, axes = plt.subplots(1,2, figsize=(11,4.8))
ax = axes[0]
ax.plot(k, mean, color='#1f78b4', linewidth=2.5, label='Overall')
ax.plot(k, easy, color='#33a02c', linewidth=2, linestyle='--', label='Easy questions')
ax.plot(k, hard, color='#e31a1c', linewidth=2, linestyle='-.', label='Hard questions')
ax.axhspan(90, 100, color='#e5f5e0', alpha=0.35)
ax.axhline(95, color='#444', linestyle=':', linewidth=1)
ax.axvline(11, color='#1f78b4', linestyle=':', linewidth=1)
ax.axvline(15, color='#e31a1c', linestyle=':', linewidth=1)
ax.annotate('K=11', (11,95), textcoords='offset points', xytext=(3,5), fontsize=8)
ax.annotate('K=15', (15,90), textcoords='offset points', xytext=(3,5), fontsize=8)
ax.set_xlabel('Number of trials (K)')
ax.set_ylabel('Consensus fidelity (%)')
ax.set_title('Reliability saturates quickly, then shows diminishing returns', fontweight='bold')
ax.legend()

ax = axes[1]
mins = [v['min_n_90'] for v in ADV['per_item_min_n'].values()]
frvals = [v['flip_rate']*100 for v in ADV['per_item_min_n'].values()]
ax.scatter(frvals, mins, s=28, alpha=0.65, color='#6a3d9a')
ax.set_xlabel('Per-item flip rate (%)')
ax.set_ylabel('Min trials for 90% fidelity')
ax.set_title('High-flip questions require many more repeats', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig8_reliability_curve.pdf', bbox_inches='tight')
plt.savefig(FIGDIR / 'fig8_reliability_curve.png', bbox_inches='tight')
plt.close()

print('All figures regenerated.')
