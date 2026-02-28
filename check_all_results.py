import json, os

for f in sorted(os.listdir('results')):
    if not f.endswith('.json'): continue
    data = json.load(open(f'results/{f}'))
    judges = set()
    ids = []
    for item in data:
        judges.add(item.get('judge_model','?'))
        ids.append(item.get('sample_id','?'))
    print(f"{f}: {len(data)} items, judges={judges}, ids={ids}")
