"""Finish remaining temp=0 trials for gpt-4.1-mini."""
import json, time, sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI
sys.stdout.reconfigure(line_buffering=True)

client = OpenAI()
pairs = json.load(open('data/eval_pairs.json'))

f = sorted(Path('results').glob('exp2_temp0_*.json'))[-1]
existing = json.load(open(f))
done_ids = {(x['judge_model'], x['sample_id']) for x in existing}

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

results = list(existing)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

for pair in pairs:
    qid = pair['id']
    if ('gpt-4.1-mini', qid) in done_ids:
        continue
    print(f"{qid}...")
    pw = []
    for i in range(10):
        try:
            prompt = PAIRWISE_PROMPT.format(question=pair['question'], response_a=pair['response_a']['text'], response_b=pair['response_b']['text'])
            resp = client.chat.completions.create(model='gpt-4.1-mini', messages=[{'role':'user','content':prompt}], temperature=0, max_tokens=512)
            content = resp.choices[0].message.content or ''
            winner = None
            if '[[A]]' in content: winner = 'A'
            elif '[[B]]' in content: winner = 'B'
            elif '[[tie]]' in content.lower(): winner = 'tie'
            pw.append({'winner': winner, 'run_index': i})
        except Exception as e:
            pw.append({'error': str(e), 'run_index': i})
        time.sleep(0.2)
    winners = [j['winner'] for j in pw if 'winner' in j and j['winner']]
    if winners:
        c = Counter(winners)
        maj = c.most_common(1)[0]
        flip = 1 - maj[1]/len(winners)
        print(f"  Flip: {flip:.0%} | {dict(c)}")
    results.append({'sample_id': qid, 'judge_model': 'gpt-4.1-mini', 'temperature': 0, 'pairwise': pw})

final = Path('results') / f'exp2_temp0_{ts}.json'
with open(final, 'w') as f2:
    json.dump(results, f2, indent=2, default=str)
print(f"Done! {len(results)} results -> {final}")
