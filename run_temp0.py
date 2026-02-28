"""
Experiment 2: Temperature=0 ablation study.
Run 10 pairwise trials per question with temp=0 to measure determinism.
"""
import json, os, sys, time, re, traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

client = OpenAI()
pairs = json.load(open('data/eval_pairs.json'))
judges = ['gpt-4o-mini', 'gpt-4.1-mini']
repeats = 10

PAIRWISE_PROMPT = """You are an impartial judge. Compare the two responses below to the given question and decide which is better.

[Question]
{question}

[Response A]
{response_a}

[Response B]
{response_b}

Evaluate based on helpfulness, relevance, accuracy, depth, and clarity.
First provide a brief explanation, then output your verdict as exactly one of:
[[A]] if Response A is better
[[B]] if Response B is better
[[tie]] if they are equally good"""

results = []
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = Path('results') / f'exp2_temp0_{ts}_partial.json'

def save():
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)

for judge_model in judges:
    print(f"\n{'='*60}")
    print(f"  Judge: {judge_model} (temp=0, {repeats} trials)")
    print(f"{'='*60}")
    
    for pair in pairs:
        qid = pair['id']
        print(f"\n  {qid}: {pair['question'][:60]}...")
        
        pw_results = []
        errors = 0
        for i in range(repeats):
            try:
                prompt = PAIRWISE_PROMPT.format(
                    question=pair['question'],
                    response_a=pair['response_a']['text'],
                    response_b=pair['response_b']['text'],
                )
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                content = resp.choices[0].message.content or ""
                winner = None
                if "[[A]]" in content: winner = "A"
                elif "[[B]]" in content: winner = "B"
                elif "[[tie]]" in content.lower(): winner = "tie"
                pw_results.append({"winner": winner, "run_index": i})
            except Exception as e:
                errors += 1
                pw_results.append({"error": str(e), "run_index": i})
                if errors >= 5:
                    print(f"    Too many errors, skipping")
                    break
            time.sleep(0.2)
        
        from collections import Counter
        winners = [j['winner'] for j in pw_results if 'winner' in j and j['winner']]
        if winners:
            c = Counter(winners)
            majority = c.most_common(1)[0]
            flip = 1 - majority[1]/len(winners)
            print(f"    Flip rate: {flip:.0%} | {dict(c)}")
        
        results.append({
            'sample_id': qid,
            'judge_model': judge_model,
            'temperature': 0,
            'pairwise': pw_results,
        })
        save()

final = Path('results') / f'exp2_temp0_{ts}.json'
with open(final, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\nDone! {len(results)} results -> {final}")
