import json
d = json.load(open('results/exp1_intra_20260226_175309_partial.json'))
item = d[0]
pw = item['pairwise']['judgments']
j0 = pw[0]
print("Keys:", list(j0.keys()))
print("Sample:", {k: str(v)[:80] for k, v in j0.items()})
