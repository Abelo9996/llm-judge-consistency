"""Resume prompt variation experiment."""
import json, time, sys
from datetime import datetime
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI
sys.stdout.reconfigure(line_buffering=True)

client = OpenAI()
pairs = json.load(open('data/eval_pairs.json'))
subset_ids = ['q001','q004','q007','q010','q013','q016','q019','q022','q025','q028']
subset = [p for p in pairs if p['id'] in subset_ids]

f = sorted(Path('results').glob('exp3_prompt_*.json'))[-1]
existing = json.load(open(f))
done = {(x['prompt_template'], x['judge_model'], x['sample_id']) for x in existing}

PROMPT_B = """Please act as a fair and unbiased evaluator. You will be given a question and two candidate answers. Your task is to determine which answer better addresses the question.

Question: {question}

Answer 1:
{response_a}

Answer 2:
{response_b}

Consider accuracy, completeness, clarity, and overall quality. Think step by step, then conclude with your final judgment in this exact format:
Winner: [[A]] or [[B]] or [[tie]]"""

results = list(existing)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
outfile = Path('results') / f'exp3_prompt_{ts}_partial.json'

def save():
    with open(outfile, 'w') as ff:
        json.dump(results, ff, indent=2, default=str)

for judge_model in ['gpt-4o-mini', 'gpt-4.1-mini']:
    for pair in subset:
        qid = pair['id']
        if ('prompt_B', judge_model, qid) in done:
            continue
        print(f"prompt_B | {judge_model} | {qid}...")
        pw = []
        for i in range(20):
            try:
                prompt = PROMPT_B.format(question=pair['question'], response_a=pair['response_a']['text'], response_b=pair['response_b']['text'])
                resp = client.chat.completions.create(model=judge_model, messages=[{"role":"user","content":prompt}], temperature=1.0, max_tokens=512)
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
            print(f"  flip={flip:.0%} | {dict(c)}")
        results.append({'sample_id': qid, 'judge_model': judge_model, 'prompt_template': 'prompt_B', 'pairwise': pw})
        save()

final = Path('results') / f'exp3_prompt_{ts}.json'
with open(final, 'w') as ff:
    json.dump(results, ff, indent=2, default=str)
print(f"Done! {len(results)} results -> {final}")
