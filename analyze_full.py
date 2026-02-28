"""Full analysis of experiment results for Paper 2."""
import json, numpy as np
from pathlib import Path
from collections import Counter

# Load latest complete results
f = sorted(Path('results').glob('exp1_v2_*.json'))[-1]
data = json.load(open(f))
print(f"Loaded {f.name}: {len(data)} items\n")

# Deduplicate: keep last occurrence per (judge, sample_id)
unique = {}
for item in data:
    key = (item['judge_model'], item['sample_id'])
    unique[key] = item
data = list(unique.values())
print(f"Unique (judge, question) pairs: {len(data)}\n")

# Analyze per judge
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    items = [d for d in data if d['judge_model'] == judge]
    print(f"\n{'='*70}")
    print(f"  JUDGE: {judge} ({len(items)} questions)")
    print(f"{'='*70}")
    
    flip_rates = []
    a_majority_count = 0
    b_majority_count = 0
    tie_count = 0
    pw_a_scores = []
    pw_b_scores = []
    pw_a_stds = []
    pw_b_stds = []
    
    for item in sorted(items, key=lambda x: x['sample_id']):
        qid = item['sample_id']
        cat = item.get('category', '')
        
        # Pairwise analysis
        pw = item.get('pairwise', [])
        if isinstance(pw, dict):
            pw = pw.get('judgments', [])
        
        winners = [j.get('winner', '?') for j in pw if isinstance(j, dict) and not j.get('error') and 'winner' in j]
        
        if winners:
            c = Counter(winners)
            majority = c.most_common(1)[0]
            flip = 1 - majority[1] / len(winners)
            flip_rates.append(flip)
            
            if majority[0] == 'A':
                a_majority_count += 1
            elif majority[0] == 'B':
                b_majority_count += 1
            else:
                tie_count += 1
        
        # Pointwise analysis
        pa = item.get('pointwise_a', [])
        pb = item.get('pointwise_b', [])
        
        a_scores = [j['score'] for j in pa if isinstance(j, dict) and not j.get('error') and j.get('score') is not None]
        b_scores = [j['score'] for j in pb if isinstance(j, dict) and not j.get('error') and j.get('score') is not None]
        
        if a_scores:
            pw_a_scores.append(np.mean(a_scores))
            pw_a_stds.append(np.std(a_scores))
        if b_scores:
            pw_b_scores.append(np.mean(b_scores))
            pw_b_stds.append(np.std(b_scores))
        
        print(f"  {qid} [{cat:12s}] flip={flip:.0%}  majority={majority[0]}({majority[1]}/{len(winners)})  "
              f"scores: A={np.mean(a_scores):.1f}±{np.std(a_scores):.2f}  B={np.mean(b_scores):.1f}±{np.std(b_scores):.2f}")
    
    print(f"\n  --- SUMMARY ---")
    print(f"  Mean flip rate: {np.mean(flip_rates):.1%} ± {np.std(flip_rates):.1%}")
    print(f"  Median flip rate: {np.median(flip_rates):.1%}")
    print(f"  Min/Max flip: {min(flip_rates):.0%} / {max(flip_rates):.0%}")
    print(f"  Position bias: A won majority {a_majority_count}/{len(items)}, B won {b_majority_count}/{len(items)}, tie {tie_count}")
    print(f"  Mean pointwise: A={np.mean(pw_a_scores):.2f}±{np.mean(pw_a_stds):.3f}  B={np.mean(pw_b_scores):.2f}±{np.mean(pw_b_stds):.3f}")
    print(f"  Score gap |A-B|: {np.mean(np.abs(np.array(pw_a_scores)-np.array(pw_b_scores))):.3f}")
    
    # Flip rate by category
    cats = {}
    for item in items:
        cat = item.get('category', 'unknown')
        pw = item.get('pairwise', [])
        if isinstance(pw, dict):
            pw = pw.get('judgments', [])
        winners = [j.get('winner', '?') for j in pw if isinstance(j, dict) and not j.get('error') and 'winner' in j]
        if winners:
            c = Counter(winners)
            majority = c.most_common(1)[0]
            flip = 1 - majority[1] / len(winners)
            cats.setdefault(cat, []).append(flip)
    
    print(f"\n  --- BY CATEGORY ---")
    for cat, flips in sorted(cats.items()):
        print(f"  {cat:20s}: mean flip={np.mean(flips):.1%} (n={len(flips)})")

# Cross-judge agreement
print(f"\n\n{'='*70}")
print(f"  CROSS-JUDGE COMPARISON")
print(f"{'='*70}")

for qid in sorted(set(d['sample_id'] for d in data)):
    items_q = {d['judge_model']: d for d in data if d['sample_id'] == qid}
    if len(items_q) < 2:
        continue
    
    majorities = {}
    flips = {}
    for judge, item in items_q.items():
        pw = item.get('pairwise', [])
        if isinstance(pw, dict):
            pw = pw.get('judgments', [])
        winners = [j.get('winner', '?') for j in pw if isinstance(j, dict) and not j.get('error') and 'winner' in j]
        if winners:
            c = Counter(winners)
            majority = c.most_common(1)[0]
            majorities[judge] = majority[0]
            flips[judge] = 1 - majority[1] / len(winners)
    
    agree = len(set(majorities.values())) == 1
    print(f"  {qid}: {'Y' if agree else 'N'} | " + " | ".join(f"{j}: {majorities[j]} (flip={flips[j]:.0%})" for j in sorted(majorities)))

agree_count = 0
total = 0
for qid in sorted(set(d['sample_id'] for d in data)):
    items_q = {d['judge_model']: d for d in data if d['sample_id'] == qid}
    if len(items_q) < 2:
        continue
    total += 1
    majorities = {}
    for judge, item in items_q.items():
        pw = item.get('pairwise', [])
        if isinstance(pw, dict):
            pw = pw.get('judgments', [])
        winners = [j.get('winner', '?') for j in pw if isinstance(j, dict) and not j.get('error') and 'winner' in j]
        if winners:
            c = Counter(winners)
            majorities[judge] = c.most_common(1)[0][0]
    if len(set(majorities.values())) == 1:
        agree_count += 1

print(f"\n  Cross-judge majority agreement: {agree_count}/{total} ({agree_count/total:.0%})")
