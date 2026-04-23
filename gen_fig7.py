"""Generate Figure 7: Prompt template sensitivity."""
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

JUDGE_LABELS = {'gpt-4o-mini': 'GPT-4o-mini', 'gpt-4.1-mini': 'GPT-4.1-mini'}
PROMPT_COLORS = {'prompt_A': '#2980b9', 'prompt_B': '#e67e22'}
PROMPT_LABELS = {'prompt_A': 'Prompt A (standard)', 'prompt_B': 'Prompt B (alternative)'}

f = sorted(Path('results').glob('exp3_prompt_*.json'))[-1]
data = json.load(open(f))

QIDS = ['q001', 'q004', 'q007', 'q010', 'q013', 'q016', 'q019', 'q022', 'q025', 'q028']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, judge in zip(axes, ['gpt-4o-mini', 'gpt-4.1-mini']):
    x = np.arange(len(QIDS))
    width = 0.35

    majority_a, majority_b = [], []
    for pt, store in [('prompt_A', majority_a), ('prompt_B', majority_b)]:
        for qid in QIDS:
            item = next((i for i in data
                         if i['judge_model'] == judge
                         and i['sample_id'] == qid
                         and i['prompt_template'] == pt), None)
            if item:
                winners = [j['winner'] for j in item['pairwise']
                           if isinstance(j, dict) and j.get('winner')]
                store.append(Counter(winners).most_common(1)[0][0] if winners else None)
            else:
                store.append(None)

    for pidx, (pt, color) in enumerate([('prompt_A', '#2980b9'), ('prompt_B', '#e67e22')]):
        vals = []
        for qid in QIDS:
            item = next((i for i in data
                         if i['judge_model'] == judge
                         and i['sample_id'] == qid
                         and i['prompt_template'] == pt), None)
            if item:
                winners = [j['winner'] for j in item['pairwise']
                           if isinstance(j, dict) and j.get('winner')]
                if winners:
                    c = Counter(winners)
                    vals.append((1 - c.most_common(1)[0][1] / len(winners)) * 100)
                else:
                    vals.append(0)
            else:
                vals.append(0)

        bars = ax.bar(x + pidx * width - width / 2, vals, width,
                      label=PROMPT_LABELS[pt], color=color, alpha=0.82,
                      edgecolor='white', zorder=3)
        for b, v in zip(bars, vals):
            if v > 3:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=7)

    # mark questions where majority outcome flipped between prompts
    for i, (ma, mb) in enumerate(zip(majority_a, majority_b)):
        if ma and mb and ma != mb:
            ax.annotate('★', xy=(i, 2), ha='center', fontsize=11,
                        color='#c0392b', zorder=5)

    ax.set_title(JUDGE_LABELS[judge], fontweight='bold')
    ax.set_ylabel('Flip Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(QIDS, rotation=45, ha='right')
    ax.set_ylim(0, 65)
    ax.legend(fontsize=8)

# Note about stars
fig.text(0.5, -0.03,
         '★ = questions where the majority-preferred response changed between prompts',
         ha='center', fontsize=9, color='#c0392b')

fig.suptitle('Prompt Template Sensitivity: Flip Rates Under Two Semantically Equivalent Prompts',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig7_prompt_sensitivity.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig7_prompt_sensitivity.png', bbox_inches='tight')
plt.close()
print("Fig 7 done")
