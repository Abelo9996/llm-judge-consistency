import json
from pathlib import Path
f = sorted(Path('results').glob('exp1_v2_*.json'))[-1]
d = json.load(open(f))
mini41 = [item['sample_id'] for item in d if item['judge_model']=='gpt-4.1-mini']
print(f"{f.name}")
print(f"gpt-4.1-mini: {len(mini41)}/29 (last: {mini41[-1] if mini41 else 'none'})")
