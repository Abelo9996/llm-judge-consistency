import json
from pathlib import Path

# Check latest partial
files = sorted(Path('results').glob('exp1_*.json'))
print("Result files:")
for f in files:
    d = json.load(open(f))
    print(f"  {f.name}: {len(d)} items")

# Check latest file in detail
latest = files[-1]
d = json.load(open(latest))
print(f"\nAnalyzing {latest.name}:")

for item in d[-3:]:
    sid = item.get('sample_id','?')
    judge = item.get('judge_model','?')
    pw = item.get('pairwise', [])
    pa = item.get('pointwise_a', [])
    pb = item.get('pointwise_b', [])
    
    pw_ok = sum(1 for j in pw if isinstance(j, dict) and j.get('error') in (None, 'None'))
    pa_ok = sum(1 for j in pa if isinstance(j, dict) and j.get('error') in (None, 'None'))
    pb_ok = sum(1 for j in pb if isinstance(j, dict) and j.get('error') in (None, 'None'))
    
    pw_errs = [j.get('error') for j in pw if isinstance(j, dict) and j.get('error') not in (None, 'None')]
    
    print(f"\n  {sid} ({judge}): PW={pw_ok}/{len(pw)} PA={pa_ok}/{len(pa)} PB={pb_ok}/{len(pb)}")
    if pw_errs:
        print(f"    Error sample: {pw_errs[0][:200]}")

# Quick API test
print("\n\nTesting API...")
from openai import OpenAI
client = OpenAI()
try:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":"Say hello"}],
        max_tokens=10
    )
    print(f"API OK: {r.choices[0].message.content}")
except Exception as e:
    print(f"API ERROR: {e}")
