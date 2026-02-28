"""
Consistency and bias metrics for LLM judge evaluation.
"""

import numpy as np
from collections import Counter
from itertools import combinations


def score_variance(scores: list[float]) -> dict:
    """Basic score distribution stats across repeated judgments."""
    scores = [s for s in scores if s is not None]
    if len(scores) < 2:
        return {"error": "insufficient scores"}
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "range": float(np.max(scores) - np.min(scores)),
        "cv": float(np.std(scores, ddof=1) / np.mean(scores)) if np.mean(scores) != 0 else None,
        "n": len(scores),
    }


def flip_rate(winners: list[str]) -> dict:
    """For pairwise judgments: how often does the winner change across repeats?"""
    valid = [w for w in winners if w is not None]
    if len(valid) < 2:
        return {"error": "insufficient judgments"}

    counts = Counter(valid)
    majority = counts.most_common(1)[0]
    flips = sum(1 for w in valid if w != majority[0])

    return {
        "flip_rate": flips / len(valid),
        "majority_winner": majority[0],
        "majority_rate": majority[1] / len(valid),
        "distribution": dict(counts),
        "n": len(valid),
    }


def intraclass_correlation(scores_matrix: list[list[float]]) -> float:
    """
    Compute ICC(3,1) — two-way mixed, single measures, consistency.
    scores_matrix: list of samples, each sample has k repeated scores.
    """
    data = np.array([s for s in scores_matrix if all(x is not None for x in s)])
    if len(data) < 2:
        return None

    n, k = data.shape
    mean_scores = data.mean(axis=1)
    grand_mean = data.mean()

    # Between-subjects sum of squares
    ss_between = k * np.sum((mean_scores - grand_mean) ** 2)

    # Within-subjects sum of squares
    ss_within = np.sum((data - mean_scores[:, np.newaxis]) ** 2)

    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    # ICC(3,1)
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    return float(icc)


def position_bias_index(scores_original: list, scores_swapped: list) -> dict:
    """
    Measure position bias by comparing scores when A/B order is swapped.
    For pairwise: winner flips suggest position bias.
    """
    assert len(scores_original) == len(scores_swapped)
    n = len(scores_original)

    # For pairwise (winners)
    if all(isinstance(s, str) for s in scores_original if s is not None):
        consistent = sum(1 for o, s in zip(scores_original, scores_swapped)
                        if o is not None and s is not None and
                        ((o == "A" and s == "B") or (o == "B" and s == "A") or (o == "tie" and s == "tie")))
        position_biased = sum(1 for o, s in zip(scores_original, scores_swapped)
                             if o is not None and s is not None and o == s and o != "tie")
        valid = sum(1 for o, s in zip(scores_original, scores_swapped)
                   if o is not None and s is not None)
        return {
            "consistent_rate": consistent / valid if valid else None,
            "position_bias_rate": position_biased / valid if valid else None,
            "n": valid,
        }

    # For pointwise (numeric scores)
    diffs = [o - s for o, s in zip(scores_original, scores_swapped)
             if o is not None and s is not None]
    if not diffs:
        return {"error": "no valid pairs"}
    return {
        "mean_score_diff": float(np.mean(diffs)),
        "std_score_diff": float(np.std(diffs)),
        "n": len(diffs),
    }


def verbosity_correlation(scores: list[float], lengths: list[int]) -> dict:
    """Pearson correlation between response length and judge score."""
    valid = [(s, l) for s, l in zip(scores, lengths) if s is not None]
    if len(valid) < 3:
        return {"error": "insufficient data"}
    s, l = zip(*valid)
    r = float(np.corrcoef(s, l)[0, 1])
    return {
        "pearson_r": r,
        "n": len(valid),
    }


def inter_judge_kappa(judgments_a: list[str], judgments_b: list[str]) -> dict:
    """Cohen's kappa between two judges on pairwise comparisons."""
    valid = [(a, b) for a, b in zip(judgments_a, judgments_b)
             if a is not None and b is not None]
    if len(valid) < 2:
        return {"error": "insufficient data"}

    a_vals, b_vals = zip(*valid)
    labels = sorted(set(a_vals) | set(b_vals))
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(valid)
    k = len(labels)

    # Confusion matrix
    matrix = np.zeros((k, k))
    for av, bv in valid:
        matrix[label_idx[av]][label_idx[bv]] += 1

    # Observed agreement
    po = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    return {
        "kappa": float(kappa),
        "observed_agreement": float(po),
        "expected_agreement": float(pe),
        "n": n,
    }
