"""
Main experiment runner for LLM-as-Judge consistency.
"""

import json
import os
import time
import random
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.evaluators.judges import judge_pairwise, judge_pointwise, judge_reference
from src.metrics.consistency import score_variance, flip_rate, intraclass_correlation, position_bias_index

console = Console()


def load_pairs(path: str = "data/eval_pairs.json") -> list:
    with open(path) as f:
        return json.load(f)


def run_exp1_intra_judge(
    pairs: list,
    judges: list[str] = ["gpt-4o-mini", "gpt-4o"],
    repeats: int = 50,
    output_dir: str = "results",
):
    """Experiment 1: Intra-judge consistency — same judge, same pair, N times."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    partial_file = Path(output_dir) / f"exp1_intra_{timestamp}_partial.json"
    os.makedirs(output_dir, exist_ok=True)

    def _save():
        with open(partial_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    for judge_model in judges:
        console.rule(f"[bold blue]Judge: {judge_model}")

        for pair in pairs:
            qid = pair["id"]
            console.print(f"  [yellow]{qid}[/yellow]: {pair['question'][:60]}...")

            # --- Pairwise ---
            pairwise_judgments = []
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                         TextColumn("{task.completed}/{task.total}"), console=console) as progress:
                ptask = progress.add_task(f"Pairwise x{repeats}", total=repeats)
                for i in range(repeats):
                    j = judge_pairwise(
                        model=judge_model,
                        question=pair["question"],
                        response_a=pair["response_a"]["text"],
                        response_b=pair["response_b"]["text"],
                        sample_id=qid,
                        run_index=i,
                    )
                    pairwise_judgments.append(j.to_dict())
                    progress.advance(ptask)
                    time.sleep(0.3)

            # --- Pointwise (both responses) ---
            pointwise_a = []
            pointwise_b = []
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                         TextColumn("{task.completed}/{task.total}"), console=console) as progress:
                ptask = progress.add_task(f"Pointwise x{repeats}", total=repeats * 2)
                for i in range(repeats):
                    ja = judge_pointwise(
                        model=judge_model, question=pair["question"],
                        response_text=pair["response_a"]["text"],
                        sample_id=f"{qid}_a", run_index=i,
                    )
                    pointwise_a.append(ja.to_dict())
                    progress.advance(ptask)
                    time.sleep(0.3)

                    jb = judge_pointwise(
                        model=judge_model, question=pair["question"],
                        response_text=pair["response_b"]["text"],
                        sample_id=f"{qid}_b", run_index=i,
                    )
                    pointwise_b.append(jb.to_dict())
                    progress.advance(ptask)
                    time.sleep(0.3)

            # Compute metrics
            pairwise_winners = [j["winner"] for j in pairwise_judgments]
            pw_scores_a = [j["score"] for j in pointwise_a]
            pw_scores_b = [j["score"] for j in pointwise_b]

            result = {
                "sample_id": qid,
                "category": pair["category"],
                "judge_model": judge_model,
                "response_a_model": pair["response_a"]["model"],
                "response_b_model": pair["response_b"]["model"],
                "pairwise": {
                    "judgments": pairwise_judgments,
                    "metrics": flip_rate(pairwise_winners),
                },
                "pointwise_a": {
                    "judgments": pointwise_a,
                    "metrics": score_variance(pw_scores_a),
                },
                "pointwise_b": {
                    "judgments": pointwise_b,
                    "metrics": score_variance(pw_scores_b),
                },
            }
            all_results.append(result)
            _save()

            # Quick summary
            fr = result["pairwise"]["metrics"]
            va = result["pointwise_a"]["metrics"]
            vb = result["pointwise_b"]["metrics"]
            if "error" not in fr:
                console.print(f"    Pairwise flip rate: {fr['flip_rate']:.1%} | "
                            f"Majority: {fr['majority_winner']} ({fr['majority_rate']:.0%})")
            if "error" not in va:
                console.print(f"    Pointwise A: {va['mean']:.1f} ± {va['std']:.2f} | "
                            f"B: {vb['mean']:.1f} ± {vb['std']:.2f}")

    # Final save
    output_file = Path(output_dir) / f"exp1_intra_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[green]Results saved to {output_file}[/green]")
    if partial_file.exists():
        partial_file.unlink()

    return all_results


def run_exp3_position_bias(
    pairs: list,
    judges: list[str] = ["gpt-4o-mini", "gpt-4o"],
    repeats: int = 30,
    output_dir: str = "results",
):
    """Experiment 3: Position bias — judge same pair with A/B swapped."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    partial_file = Path(output_dir) / f"exp3_bias_{timestamp}_partial.json"
    os.makedirs(output_dir, exist_ok=True)

    def _save():
        with open(partial_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    for judge_model in judges:
        console.rule(f"[bold blue]Judge: {judge_model} (Position Bias)")

        for pair in pairs:
            qid = pair["id"]
            console.print(f"  [yellow]{qid}[/yellow]: {pair['question'][:60]}...")

            original_winners = []
            swapped_winners = []

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                         TextColumn("{task.completed}/{task.total}"), console=console) as progress:
                ptask = progress.add_task(f"Bias test x{repeats}", total=repeats * 2)

                for i in range(repeats):
                    # Original order
                    j1 = judge_pairwise(
                        model=judge_model, question=pair["question"],
                        response_a=pair["response_a"]["text"],
                        response_b=pair["response_b"]["text"],
                        sample_id=f"{qid}_orig", run_index=i,
                    )
                    original_winners.append(j1.winner)
                    progress.advance(ptask)
                    time.sleep(0.3)

                    # Swapped order
                    j2 = judge_pairwise(
                        model=judge_model, question=pair["question"],
                        response_a=pair["response_b"]["text"],
                        response_b=pair["response_a"]["text"],
                        sample_id=f"{qid}_swap", run_index=i,
                    )
                    swapped_winners.append(j2.winner)
                    progress.advance(ptask)
                    time.sleep(0.3)

            bias_metrics = position_bias_index(original_winners, swapped_winners)
            result = {
                "sample_id": qid,
                "category": pair["category"],
                "judge_model": judge_model,
                "original_winners": original_winners,
                "swapped_winners": swapped_winners,
                "bias_metrics": bias_metrics,
            }
            all_results.append(result)
            _save()

            if "error" not in bias_metrics:
                console.print(f"    Position bias rate: {bias_metrics['position_bias_rate']:.1%} | "
                            f"Consistent: {bias_metrics['consistent_rate']:.1%}")

    output_file = Path(output_dir) / f"exp3_bias_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[green]Results saved to {output_file}[/green]")
    if partial_file.exists():
        partial_file.unlink()
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["exp1", "exp3", "all"], default="all")
    parser.add_argument("--judges", nargs="+", default=["gpt-4o-mini", "gpt-4o"])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--data", default="data/eval_pairs.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    pairs = load_pairs(args.data)

    if args.experiment in ("exp1", "all"):
        run_exp1_intra_judge(pairs, args.judges, args.repeats, args.output)
    if args.experiment in ("exp3", "all"):
        run_exp3_position_bias(pairs, args.judges, min(args.repeats, 30), args.output)
