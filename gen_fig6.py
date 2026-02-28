"""Generate Figure 6: Temperature ablation comparison."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

plt.rcParams.update({'font.size': 11, 'figure.dpi': 300})

def get_flip_rates(filepath, judge):
    d = json.load(open(filepath))
    items = [x for x in d if x['judge_model']==judge]
    result = {}
    for item in sorted(items, key=lambda x: x['sample_id']):
        pw = item.get('pairwise', [])
        if isinstance(pw, dict): pw = pw.get('judgments', [])
        winners = [j.get('winner','?') for j in pw if isinstance(j,dict) and not j.get('error') and j.get('winner')]
        if winners:
            c = Counter(winners)
            fr = 1 - c.most_common(1)[0][1]/len(winners)
            result[item['sample_id']] = fr
    return result

# Load temp=1 and temp=0 data
f1 = sorted(Path('results').glob('exp1_v2_*.json'))[-1]
f0 = sorted(Path('results').glob('exp2_temp0_*.json'))[-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, judge in enumerate(['gpt-4o-mini', 'gpt-4.1-mini']):
    t1 = get_flip_rates(f1, judge)
    t0 = get_flip_rates(f0, judge)
    
    qids = sorted(set(t1.keys()) & set(t0.keys()))
    x = np.arange(len(qids))
    w = 0.35
    
    vals_t1 = [t1[q]*100 for q in qids]
    vals_t0 = [t0[q]*100 for q in qids]
    
    axes[idx].bar(x - w/2, vals_t1, w, label='temp=1.0', color='#e74c3c', alpha=0.8)
    axes[idx].bar(x + w/2, vals_t0, w, label='temp=0', color='#3498db', alpha=0.8)
    axes[idx].set_title(f'{judge}', fontweight='bold')
    axes[idx].set_ylabel('Flip Rate (%)')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(qids, rotation=45, ha='right', fontsize=6)
    axes[idx].legend(fontsize=9)
    axes[idx].set_ylim(0, 60)

plt.suptitle('Temperature Ablation: Flip Rates at temp=1.0 vs temp=0', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig6_temp_ablation.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig6_temp_ablation.png', bbox_inches='tight')
plt.close()
print("Fig 6 done")

# Summary stats
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    t1 = get_flip_rates(f1, judge)
    t0 = get_flip_rates(f0, judge)
    print(f"\n{judge}:")
    print(f"  temp=1: mean={np.mean(list(t1.values()))*100:.1f}%")
    print(f"  temp=0: mean={np.mean(list(t0.values()))*100:.1f}%")
    print(f"  reduction: {(1-np.mean(list(t0.values()))/np.mean(list(t1.values())))*100:.0f}%")
