import json

with open('results/exp1_intra_20260225_224253_partial.json') as f:
    results = json.load(f)

print(f'Paper 2 partial results: {len(results)} sample-judge combos')
judges = set(r['judge_model'] for r in results)
print(f'Judges: {judges}')

by_judge = {}
for r in results:
    j = r['judge_model']
    by_judge.setdefault(j, []).append(r)

for judge, items in by_judge.items():
    print(f'\n=== {judge}: {len(items)} samples ===')
    for r in items[:8]:
        fr = r['pairwise']['metrics']
        va = r['pointwise_a']['metrics']
        vb = r['pointwise_b']['metrics']
        flip = f"{fr['flip_rate']:.1%}" if 'flip_rate' in fr else 'ERR'
        maj = fr.get('majority_winner', '?')
        std_a = f"{va['std']:.2f}" if 'std' in va else 'ERR'
        std_b = f"{vb['std']:.2f}" if 'std' in vb else 'ERR'
        mean_a = f"{va['mean']:.1f}" if 'mean' in va else '?'
        mean_b = f"{vb['mean']:.1f}" if 'mean' in vb else '?'
        print(f'  {r["sample_id"]:6} flip={flip} maj={maj} | ptwise A={mean_a}+/-{std_a} B={mean_b}+/-{std_b}')
    if len(items) > 8:
        print(f'  ... and {len(items)-8} more')
