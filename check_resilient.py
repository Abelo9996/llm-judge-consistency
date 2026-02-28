import json, os
for f in os.listdir('results'):
    if 'resilient' in f:
        data = json.load(open(f'results/{f}'))
        print(f'{f} ({len(data)} items):')
        for d in data:
            pw_ok = sum(1 for j in d['pairwise']['judgments'] if 'error' not in j)
            print(f"  {d['sample_id']} ({d['judge_model']}): pw={pw_ok}/50")
