"""
Statistical significance tests for paper findings.
"""
import json, numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats

# Load data
f = sorted(Path('results').glob('exp1_v2_*.json'))[-1]
data = json.load(open(f))
unique = {}
for item in data:
    unique[(item['judge_model'], item['sample_id'])] = item
data = list(unique.values())

def get_flip_rate(item):
    pw = item.get('pairwise', [])
    if isinstance(pw, dict): pw = pw.get('judgments', [])
    winners = [j.get('winner','?') for j in pw if isinstance(j,dict) and not j.get('error') and 'winner' in j]
    if not winners: return None
    c = Counter(winners)
    return 1 - c.most_common(1)[0][1] / len(winners)

print("="*60)
print("STATISTICAL TESTS")
print("="*60)

# 1. Binomial test: is position bias significant per question?
print("\n--- Position Bias: Binomial Tests ---")
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    items = [d for d in data if d['judge_model']==judge]
    sig_count = 0
    for item in items:
        pw = item.get('pairwise', [])
        if isinstance(pw, dict): pw = pw.get('judgments', [])
        winners = [j.get('winner','?') for j in pw if isinstance(j,dict) and not j.get('error') and 'winner' in j]
        n_a = sum(1 for w in winners if w == 'A')
        n = len(winners)
        if n > 0:
            p = stats.binomtest(n_a, n, 0.5)
            if p.pvalue < 0.05:
                sig_count += 1
    print(f"  {judge}: {sig_count}/29 questions have significant position bias (p<0.05)")

# 2. Aggregate position bias: sign test
print("\n--- Aggregate Position Bias: Sign Test ---")
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    items = [d for d in data if d['judge_model']==judge]
    a_wins = 0
    for item in items:
        pw = item.get('pairwise', [])
        if isinstance(pw, dict): pw = pw.get('judgments', [])
        winners = [j.get('winner','?') for j in pw if isinstance(j,dict) and not j.get('error') and 'winner' in j]
        n_a = sum(1 for w in winners if w == 'A')
        if n_a > len(winners)/2:
            a_wins += 1
    p = stats.binomtest(a_wins, 29, 0.5)
    print(f"  {judge}: A won majority in {a_wins}/29 questions, p={p.pvalue:.6f}")

# 3. Mann-Whitney U: flip rates between judges
print("\n--- Flip Rate Comparison: Mann-Whitney U ---")
flips_4o = [get_flip_rate(d) for d in data if d['judge_model']=='gpt-4o-mini' and get_flip_rate(d) is not None]
flips_41 = [get_flip_rate(d) for d in data if d['judge_model']=='gpt-4.1-mini' and get_flip_rate(d) is not None]
u, p = stats.mannwhitneyu(flips_4o, flips_41, alternative='two-sided')
print(f"  GPT-4o-mini mean FR: {np.mean(flips_4o):.3f}, GPT-4.1-mini mean FR: {np.mean(flips_41):.3f}")
print(f"  U={u:.1f}, p={p:.4f}")

# 4. Kruskal-Wallis: flip rates across categories
print("\n--- Category Differences: Kruskal-Wallis ---")
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    items = [d for d in data if d['judge_model']==judge]
    cat_flips = {}
    for item in items:
        cat = item.get('category', 'unknown')
        fr = get_flip_rate(item)
        if fr is not None:
            cat_flips.setdefault(cat, []).append(fr)
    groups = [v for v in cat_flips.values() if len(v) > 1]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        print(f"  {judge}: H={h:.2f}, p={p:.4f}")

# 5. Paired Wilcoxon: pointwise A vs B scores
print("\n--- Pointwise A vs B: Wilcoxon Signed-Rank ---")
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    items = [d for d in data if d['judge_model']==judge]
    a_means = []
    b_means = []
    for item in items:
        pa = item.get('pointwise_a', [])
        pb = item.get('pointwise_b', [])
        a_scores = [j['score'] for j in pa if isinstance(j,dict) and not j.get('error') and j.get('score') is not None]
        b_scores = [j['score'] for j in pb if isinstance(j,dict) and not j.get('error') and j.get('score') is not None]
        if a_scores and b_scores:
            a_means.append(np.mean(a_scores))
            b_means.append(np.mean(b_scores))
    if a_means:
        w, p = stats.wilcoxon(a_means, b_means)
        print(f"  {judge}: mean A={np.mean(a_means):.3f}, mean B={np.mean(b_means):.3f}, W={w:.1f}, p={p:.4f}")

# 6. Cohen's kappa for cross-judge agreement
print("\n--- Cross-Judge Agreement: Cohen's Kappa ---")
labels_4o = []
labels_41 = []
for qid in sorted(set(d['sample_id'] for d in data)):
    items_q = {d['judge_model']: d for d in data if d['sample_id'] == qid}
    if len(items_q) < 2: continue
    for judge, item in items_q.items():
        pw = item.get('pairwise', [])
        if isinstance(pw, dict): pw = pw.get('judgments', [])
        winners = [j.get('winner','?') for j in pw if isinstance(j,dict) and not j.get('error') and 'winner' in j]
        if winners:
            c = Counter(winners)
            maj = c.most_common(1)[0][0]
            if judge == 'gpt-4o-mini': labels_4o.append(maj)
            else: labels_41.append(maj)

if labels_4o and labels_41:
    kappa = stats.contingency.association(
        np.array([[sum(1 for a,b in zip(labels_4o,labels_41) if a==x and b==y) 
                   for y in ['A','B','tie']] for x in ['A','B','tie']]),
        method='cramer'
    )
    # Manual Cohen's kappa
    n = len(labels_4o)
    agree = sum(1 for a,b in zip(labels_4o,labels_41) if a==b)
    po = agree/n
    # Expected agreement
    cats = set(labels_4o + labels_41)
    pe = sum((labels_4o.count(c)/n) * (labels_41.count(c)/n) for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    print(f"  Observed agreement: {po:.3f} ({agree}/{n})")
    print(f"  Expected agreement: {pe:.3f}")
    print(f"  Cohen's kappa: {kappa:.3f}")

# 7. Effect size: mean flip rate confidence intervals (bootstrap)
print("\n--- Flip Rate 95% CI (Bootstrap) ---")
rng = np.random.default_rng(42)
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    flips = [get_flip_rate(d) for d in data if d['judge_model']==judge and get_flip_rate(d) is not None]
    boot_means = [np.mean(rng.choice(flips, len(flips), replace=True)) for _ in range(10000)]
    ci = np.percentile(boot_means, [2.5, 97.5])
    print(f"  {judge}: mean FR = {np.mean(flips):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
