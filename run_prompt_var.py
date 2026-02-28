"""
Experiment 3: Prompt template sensitivity.
Run 2 different prompt templates on 10 questions x 20 trials each.
"""
import json, time, sys, re
from datetime import datetime
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI
sys.stdout.reconfigure(line_buffering=True)

client = OpenAI()
pairs = json.load(open('data/eval_pairs.json'))

# Use 10 diverse questions
subset_ids = ['q001','q004','q007','q010','q013','q016','q019','q022','q025','q028']
subset = [p for p in pairs if p['id'] in subset_ids]

PROMPT_A = """You are an impartial judge. Compare the two responses below to the given question and decide which is better.

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

PROMPT_B = """Please act as a fair and unbiased evaluator. You will be given a question and two candidate answers. Your task is to determine which answer better addresses the question.

Question: {question}

Answer 1:
{response_a}

Answer 2:
{response_b}

Consider accuracy, completeness, clarity, and overall quality. Think step by step, then conclude with your final judgment in this exact format:
Winner: [[A]] or [[B]] or [[tie]]"""

results = []
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
outfile = Path('results') / f'exp3_prompt_{ts}_partial.json'
repeats = 20

def save():
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)

for prompt_name, template in [('prompt_A', PROMPT_A), ('prompt_B', PROMPT_B)]:
    for judge_model in ['gpt-4o-mini', 'gpt-4.1-mini']:
        print(f"\n{'='*60}")
        print(f"  {prompt_name} | {judge_model}")
        print(f"{'='*60}")
        
        for pair in subset:
            qid = pair['id']
            print(f"  {qid}...", end=' ')
            
            pw = []
            for i in range(repeats):
                try:
                    prompt = template.format(
                        question=pair['question'],
                        response_a=pair['response_a']['text'],
                        response_b=pair['response_b']['text'],
                    )
                    resp = client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        max_tokens=512,
                    )
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
                print(f"flip={flip:.0%} | {dict(c)}")
            
            results.append({
                'sample_id': qid,
                'judge_model': judge_model,
                'prompt_template': prompt_name,
                'pairwise': pw,
            })
            save()

final = Path('results') / f'exp3_prompt_{ts}.json'
with open(final, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nDone! {len(results)} results -> {final}")
