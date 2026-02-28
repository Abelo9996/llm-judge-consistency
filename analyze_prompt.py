"""Analyze prompt sensitivity experiment."""
import json, numpy as np
from pathlib import Path
from collections import Counter

f = sorted(Path('results').glob('exp3_prompt_*.json'))[-1]
d = json.load(open(f))

print('PROMPT SENSITIVITY ANALYSIS')
print('='*60)

for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    print(f'\n{judge}:')
    for qid in ['q001','q004','q007','q010','q013','q016','q019','q022','q025','q028']:
        row = []
        for pt in ['prompt_A','prompt_B']:
            item = next((x for x in d if x['judge_model']==judge and x['sample_id']==qid and x['prompt_template']==pt), None)
            if item:
                winners = [j['winner'] for j in item['pairwise'] if 'winner' in j and j['winner']]
                if winners:
                    c = Counter(winners)
                    maj = c.most_common(1)[0]
                    row.append((maj[0], 1-maj[1]/len(winners)))
                else:
                    row.append(('?', 0))
            else:
                row.append(('?', 0))
        agree = row[0][0] == row[1][0]
        tag = 'AGREE' if agree else 'DISAGREE'
        print(f'  {qid}: pA={row[0][0]}({row[0][1]:.0%}) pB={row[1][0]}({row[1][1]:.0%}) {tag}')

agree_count = 0
total = 0
flip_changes = []
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    for qid in ['q001','q004','q007','q010','q013','q016','q019','q022','q025','q028']:
        items = {x['prompt_template']: x for x in d if x['judge_model']==judge and x['sample_id']==qid}
        if len(items) < 2: continue
        total += 1
        majs = {}
        frs = {}
        for pt, item in items.items():
            winners = [j['winner'] for j in item['pairwise'] if 'winner' in j and j['winner']]
            if winners:
                c = Counter(winners)
                majs[pt] = c.most_common(1)[0][0]
                frs[pt] = 1 - c.most_common(1)[0][1]/len(winners)
        if len(set(majs.values())) == 1:
            agree_count += 1
        if len(frs) == 2:
            flip_changes.append(abs(list(frs.values())[0] - list(frs.values())[1]))

print(f'\nCross-prompt agreement: {agree_count}/{total} ({agree_count/total:.0%})')
print(f'Mean flip rate change between prompts: {np.mean(flip_changes):.1%}')
