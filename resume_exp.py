"""Resume experiment 1 from partial results, then run exp3."""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Load partial results to find where we left off
partial = "results/exp1_intra_20260226_175309_partial.json"
with open(partial) as f:
    done = json.load(f)

done_ids = {r["sample_id"] for r in done}
done_judges = {r["judge_model"] for r in done}
print(f"Already completed: {len(done)} items")
print(f"  IDs: {sorted(done_ids)}")
print(f"  Judges: {done_judges}")

# Load eval pairs
with open("data/eval_pairs.json") as f:
    all_pairs = json.load(f)

remaining_pairs = [p for p in all_pairs if p["id"] not in done_ids]
print(f"Remaining pairs: {len(remaining_pairs)} (of {len(all_pairs)})")

# Now run the remaining via the existing runner, but we patch it to append to our results
from src.runners.run_experiment import run_exp1_intra_judge, run_exp3_position_bias

judges = ["gpt-4o-mini", "gpt-4.1-mini"]

# Run exp1 on remaining pairs for gpt-4o-mini, then all pairs for gpt-4.1-mini
if remaining_pairs:
    print(f"\n=== Resuming exp1 for gpt-4o-mini: {len(remaining_pairs)} pairs ===")
    new_results = run_exp1_intra_judge(remaining_pairs, ["gpt-4o-mini"], repeats=50)
    # Merge with partial
    done.extend(new_results)
    
# Now run gpt-4.1-mini on ALL pairs (none completed yet)
print(f"\n=== Running exp1 for gpt-4.1-mini: {len(all_pairs)} pairs ===")
mini_results = run_exp1_intra_judge(all_pairs, ["gpt-4.1-mini"], repeats=50)
done.extend(mini_results)

# Save merged exp1
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"results/exp1_intra_{ts}_merged.json"
with open(outfile, "w") as f:
    json.dump(done, f, indent=2, default=str)
print(f"\nExp1 merged results saved to {outfile}")

# Now run exp3
print(f"\n=== Running exp3 (position bias) ===")
run_exp3_position_bias(all_pairs, judges, repeats=30)

print("\n=== ALL EXPERIMENTS COMPLETE ===")
