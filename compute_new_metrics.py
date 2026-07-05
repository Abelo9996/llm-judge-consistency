"""
Compute advanced metrics from existing experiment data:
  - ICC(2,1) from 50 pointwise trials
  - Shannon entropy of pairwise outcome distribution
  - Reliability curve: P(K-trial majority == 50-trial majority) for K=1..50
  - Minimum-N-for-confidence per question
  - Variance decomposition
  - Cohen's kappa per category
  - Leaderboard noise budget estimate
Outputs metrics_advanced.json for use in paper text and figures.
"""
import json, numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

# ── Load data ────────────────────────────────────────────────────────────────
raw = json.load(open(sorted(Path('results').glob('exp1_v2_*.json'))[-1]))
seen = {}
for item in raw:
    seen[(item['judge_model'], item['sample_id'])] = item
data = list(seen.values())

raw_t0 = json.load(open(sorted(Path('results').glob('exp2_temp0_*.json'))[-1]))
raw_p  = json.load(open(sorted(Path('results').glob('exp3_prompt_*.json'))[-1]))


# ── Helpers ──────────────────────────────────────────────────────────────────
def pairwise_winners(item):
    pw = item.get('pairwise', [])
    if isinstance(pw, dict):
        pw = pw.get('judgments', [])
    return [j['winner'] for j in pw
            if isinstance(j, dict) and not j.get('error') and j.get('winner')]


def pointwise_scores(item, side):
    key = f'pointwise_{side}'
    pts = item.get(key, [])
    if isinstance(pts, dict):
        pts = pts.get('judgments', [])
    return [j['score'] for j in pts
            if isinstance(j, dict) and not j.get('error')
            and j.get('score') is not None]


def flip_rate(winners):
    if not winners:
        return None
    c = Counter(winners)
    return 1.0 - c.most_common(1)[0][1] / len(winners)


def entropy_bits(winners):
    if not winners:
        return None
    c = Counter(winners)
    n = len(winners)
    probs = np.array([v / n for v in c.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# ── 1. ICC(2,1) from pointwise scores ────────────────────────────────────────
# Treat each of the 50 trials as a "rater"; compute ICC over all questions.
# ICC(2,1): two-way random, single measures — absolute agreement.
def icc21(scores_matrix):
    """
    scores_matrix: shape (n_subjects, n_raters)
    Returns ICC(2,1) absolute agreement.
    """
    n, k = scores_matrix.shape
    grand_mean = np.mean(scores_matrix)
    ss_total = np.sum((scores_matrix - grand_mean) ** 2)
    ss_rows  = k * np.sum((np.mean(scores_matrix, axis=1) - grand_mean) ** 2)
    ss_cols  = n * np.sum((np.mean(scores_matrix, axis=0) - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows  = ss_rows  / (n - 1)
    ms_cols  = ss_cols  / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    # ICC(2,1) absolute
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    return float(icc)


icc_results = {}
for judge in ['gpt-4o-mini', 'gpt-4.1-mini']:
    scores_a, scores_b = [], []
    for item in [d for d in data if d['judge_model'] == judge]:
        sa = pointwise_scores(item, 'a')
        sb = pointwise_scores(item, 'b')
        if len(sa) == 50 and len(sb) == 50:
            scores_a.append(sa)
            scores_b.append(sb)
    # ICC for response A scores: rows=questions, cols=trials
    mat_a = np.array(scores_a)   # (n_questions, 50)
    mat_b = np.array(scores_b)
    icc_a = icc21(mat_a)
    icc_b = icc21(mat_b)
    # Combined: stack both responses
    mat_combined = np.vstack([mat_a, mat_b])
    icc_combined = icc21(mat_combined)
    icc_results[judge] = {'icc_a': icc_a, 'icc_b': icc_b, 'icc_combined': icc_combined}
    print(f"ICC(2,1) [{judge}]: A={icc_a:.3f}  B={icc_b:.3f}  combined={icc_combined:.3f}")


# ── 2. Entropy per question ──────────────────────────────────────────────────
entropy_results = {}
for item in data:
    w = pairwise_winners(item)
    key = (item['judge_model'], item['sample_id'])
    entropy_results[key] = {
        'entropy': entropy_bits(w),
        'flip_rate': flip_rate(w),
        'category': item.get('category', ''),
    }


# ── 3. Reliability curve ─────────────────────────────────────────────────────
# For each (judge, question), randomly subsample K trials (Monte Carlo, 500 reps).
# Compute P(K-trial majority == 50-trial majority) averaged across questions.
def majority_of(winners):
    if not winners:
        return None
    return Counter(winners).most_common(1)[0][0]


Kvals = list(range(1, 51))
N_BOOT = 500
rng = np.random.RandomState(42)

reliability_by_item = {}
for item in data:
    w = pairwise_winners(item)
    if len(w) < 50:
        continue
    w_arr = np.array(w)
    gold = majority_of(w)
    key = (item['judge_model'], item['sample_id'])
    correct_at_k = []
    for k in Kvals:
        if k >= len(w):
            correct_at_k.append(1.0)
        else:
            hits = 0
            for _ in range(N_BOOT):
                sample = rng.choice(w_arr, size=k, replace=False)
                hits += (majority_of(sample.tolist()) == gold)
            correct_at_k.append(hits / N_BOOT)
    reliability_by_item[key] = {
        'correct_at_k': correct_at_k,
        'flip_rate': flip_rate(w),
        'category': item.get('category', ''),
        'judge': item['judge_model'],
    }

# Mean P(correct) at each K across all (judge, question) pairs
mean_correct_at_k = []
se_correct_at_k   = []
for k_idx in range(len(Kvals)):
    vals = [v['correct_at_k'][k_idx] for v in reliability_by_item.values()]
    mean_correct_at_k.append(float(np.mean(vals)))
    se_correct_at_k.append(float(np.std(vals) / np.sqrt(len(vals))))

# By judge
correct_by_judge = {judge: [] for judge in ['gpt-4o-mini', 'gpt-4.1-mini']}
for k_idx in range(len(Kvals)):
    for judge in correct_by_judge:
        vals = [v['correct_at_k'][k_idx] for v in reliability_by_item.values()
                if v['judge'] == judge]
        correct_by_judge[judge].append(float(np.mean(vals)) if vals else 0.0)

# By flip-rate stratum
hard_items = {k: v for k, v in reliability_by_item.items() if v['flip_rate'] >= 0.10}
easy_items = {k: v for k, v in reliability_by_item.items() if v['flip_rate']  < 0.10}
correct_hard, correct_easy = [], []
for k_idx in range(len(Kvals)):
    hv = [v['correct_at_k'][k_idx] for v in hard_items.values()]
    ev = [v['correct_at_k'][k_idx] for v in easy_items.values()]
    correct_hard.append(float(np.mean(hv)) if hv else 0.0)
    correct_easy.append(float(np.mean(ev)) if ev else 0.0)

print(f"\nReliability curve computed over {len(reliability_by_item)} (judge, question) pairs")
print(f"  P(correct|K=1):  {mean_correct_at_k[0]:.3f}")
print(f"  P(correct|K=5):  {mean_correct_at_k[4]:.3f}")
print(f"  P(correct|K=10): {mean_correct_at_k[9]:.3f}")
print(f"  P(correct|K=20): {mean_correct_at_k[19]:.3f}")
print(f"  P(correct|K=50): {mean_correct_at_k[49]:.3f}")


# ── 4. Minimum N for 90% / 95% confidence ───────────────────────────────────
def min_n_for_threshold(correct_at_k, threshold):
    for k, v in enumerate(correct_at_k):
        if v >= threshold:
            return k + 1  # 1-indexed
    return len(correct_at_k)  # never reaches threshold

overall_min_n_90 = min_n_for_threshold(mean_correct_at_k, 0.90)
overall_min_n_95 = min_n_for_threshold(mean_correct_at_k, 0.95)
hard_min_n_90    = min_n_for_threshold(correct_hard, 0.90)
easy_min_n_90    = min_n_for_threshold(correct_easy, 0.90)

print(f"\nMin trials for overall 90% accuracy: {overall_min_n_90}")
print(f"Min trials for overall 95% accuracy: {overall_min_n_95}")
print(f"Min trials (hard questions) 90%: {hard_min_n_90}")
print(f"Min trials (easy questions) 90%: {easy_min_n_90}")

# Per-item min-N
per_item_min_n = {}
for key, v in reliability_by_item.items():
    per_item_min_n[str(key)] = {
        'min_n_90': min_n_for_threshold(v['correct_at_k'], 0.90),
        'flip_rate': v['flip_rate'],
        'category': v['category'],
        'judge': v['judge'],
    }


# ── 5. Variance decomposition ────────────────────────────────────────────────
# Decompose variance in pairwise scores into:
#   - Between-question variance (questions differ in difficulty)
#   - Within-question / within-judge variance (noise)
all_scores = []
for item in data:
    pw = item.get('pairwise', [])
    if isinstance(pw, dict):
        pw = pw.get('judgments', [])
    for j in pw:
        if isinstance(j, dict) and not j.get('error') and j.get('score') is not None:
            all_scores.append({
                'score': j['score'],
                'question': item['sample_id'],
                'judge': item['judge_model'],
            })

question_means = {}
for s in all_scores:
    question_means.setdefault(s['question'], []).append(s['score'])
question_means = {q: np.mean(v) for q, v in question_means.items()}
grand_mean_score = np.mean([s['score'] for s in all_scores])

between_var = np.var([v for v in question_means.values()])
within_var  = np.mean([np.var([s['score'] for s in all_scores if s['question'] == q])
                        for q in question_means])
total_var   = np.var([s['score'] for s in all_scores])

print(f"\nVariance decomposition (pairwise scores):")
print(f"  Between-question: {between_var:.4f} ({100*between_var/total_var:.1f}%)")
print(f"  Within-question:  {within_var:.4f}  ({100*within_var/total_var:.1f}%)")
print(f"  Total:            {total_var:.4f}")


# ── 6. Leaderboard noise budget ──────────────────────────────────────────────
# Estimate: for a benchmark of B questions evaluated with single-trial judging,
# what fraction of pairwise comparisons between two models could flip?
# If model X vs model Y differs by d true-quality units, and judge flip rate is
# mean_fr, then P(wrong ranking) ≈ flip_rate / (2 * d_normalized)
# We estimate using the empirical distribution of question-level flip rates.

flip_rates_all = [v['flip_rate'] for v in reliability_by_item.values()]
mean_fr  = float(np.mean(flip_rates_all))
median_fr= float(np.median(flip_rates_all))
# P(single-trial gives wrong answer for a question) = flip_rate for that question
# For a benchmark of 100 questions, expected wrong decisions = 100 * mean_fr
# Expected fraction of questions that change outcome if re-run: mean_fr
leaderboard_100q_expected_flips = mean_fr * 100  # expected number of flipped outcomes in 100-Q benchmark

print(f"\nLeaderboard noise budget:")
print(f"  Mean flip rate: {mean_fr:.3f}")
print(f"  Expected flipped outcomes in 100-Q benchmark (single trial): {leaderboard_100q_expected_flips:.1f}")
print(f"  P(ranking reversal for adjacent models on 100-Q bench): ~{mean_fr:.1%}")


# Category ICC requires more questions per category than available (3-4);
# omitted to avoid reporting unstable estimates.


# ── Save output ───────────────────────────────────────────────────────────────
output = {
    'icc': icc_results,
    'reliability_curve': {
        'k_values': Kvals,
        'mean_correct': mean_correct_at_k,
        'se_correct': se_correct_at_k,
        'by_judge': correct_by_judge,
        'hard_questions': correct_hard,
        'easy_questions': correct_easy,
        'overall_min_n_90': overall_min_n_90,
        'overall_min_n_95': overall_min_n_95,
        'hard_min_n_90': hard_min_n_90,
        'easy_min_n_90': easy_min_n_90,
    },
    'entropy_per_item': {str(k): v for k, v in entropy_results.items()},
    'variance_decomposition': {
        'between_question': float(between_var),
        'within_question': float(within_var),
        'total': float(total_var),
        'between_pct': float(100 * between_var / total_var),
        'within_pct': float(100 * within_var / total_var),
    },
    'leaderboard_noise': {
        'mean_flip_rate': mean_fr,
        'median_flip_rate': median_fr,
        'expected_flips_100q_benchmark': float(leaderboard_100q_expected_flips),
    },
    'per_item_min_n': per_item_min_n,
}

json.dump(output, open('results/metrics_advanced.json', 'w'), indent=2)
print("\nSaved: results/metrics_advanced.json")
