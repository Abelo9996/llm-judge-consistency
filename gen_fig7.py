"""Generate Figure 7: Prompt template sensitivity."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

plt.rcParams.update({'font.size': 11, 'figure.dpi': 300})

f = sorted(Path('results').glob('exp3_prompt_*.json'))[-1]
d = json.load(open(f))

qids = ['q001','q004','q007','q010','q013','q016','q019','q022','q025','q028']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, judge in enumerate(['gpt-4o-mini', 'gpt-4.1-mini']):
    x = np.arange(len(qids))
    w = 0.35
    for pidx, (pt, color, label) in enumerate([('prompt_A','#e74c3c','Prompt A (standard)'), ('prompt_B','#3498db','Prompt B (alternative)')]):
        vals = []
        for qid in qids:
            item = next((i for i in d if i['judge_model']==judge and i['sample_id']==qid and i['prompt_template']==pt), None)
            if item:
                winners = [j['winner'] for j in item['pairwise'] if 'winner' in j and j['winner']]
                if winners:
                    c = Counter(winners)
                    vals.append((1 - c.most_common(1)[0][1]/len(winners))*100)
                else:
                    vals.append(0)
            else:
                vals.append(0)
        axes[idx].bar(x + pidx*w - w/2, vals, w, label=label, color=color, alpha=0.8)
    
    axes[idx].set_title(f'{judge}', fontweight='bold')
    axes[idx].set_ylabel('Flip Rate (%)')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(qids, rotation=45, ha='right')
    axes[idx].legend(fontsize=8)
    axes[idx].set_ylim(0, 60)

plt.suptitle('Prompt Template Sensitivity: Flip Rates Across Two Prompt Designs', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig7_prompt_sensitivity.pdf', bbox_inches='tight')
plt.savefig('paper/figures/fig7_prompt_sensitivity.png', bbox_inches='tight')
plt.close()
print("Fig 7 done")
