import json
f = 'results/exp1_intra_20260226_175309_partial.json'
data = json.load(open(f))
print(f"13 items, sample_ids: {[d['sample_id'] for d in data]}")
print(f"Judge model: {data[0]['judge_model']}")
print(f"Pairwise trials in first item: {len(data[0]['pairwise'])}")
print(f"Pointwise_a trials in first item: {len(data[0]['pointwise_a'])}")
