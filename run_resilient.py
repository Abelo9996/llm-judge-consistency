"""
Resilient experiment runner v2 — properly reads old partials, catches errors, continues.
"""
import json, os, sys, time, traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

def get_good_results():
    """Read all partial files and extract only items with actual data (>0 successful judgments)."""
    good = {}  # (judge, sample_id) -> item
    for f in sorted(Path('results').glob('exp1_*.json')):
        data = json.load(open(f))
        for item in data:
            judge = item.get('judge_model', '')
            sid = item.get('sample_id', '')
            # Check if this has real data
            pw = item.get('pairwise', [])
            if isinstance(pw, list):
                ok = sum(1 for j in pw if isinstance(j, dict) and not j.get('error') and 'winner' in j)
            elif isinstance(pw, dict) and 'judgments' in pw:
                ok = sum(1 for j in pw['judgments'] if isinstance(j, dict) and not j.get('error'))
            else:
                ok = 0
            if ok >= 25:  # at least half succeeded
                good[(judge, sid)] = item
                print(f"  Loaded {sid} ({judge}): {ok} good pairwise")
    return good

def main():
    sys.stdout.reconfigure(line_buffering=True)
    
    from src.evaluators.judges import judge_pairwise, judge_pointwise
    
    pairs = json.load(open('data/eval_pairs.json'))
    judges = ['gpt-4o-mini', 'gpt-4.1-mini']
    repeats = 50
    
    print("Loading existing results...")
    good = get_good_results()
    print(f"Found {len(good)} good results\n")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = list(good.values())
    outfile = Path('results') / f'exp1_v2_{ts}_partial.json'
    
    def save():
        with open(outfile, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    for judge_model in judges:
        done_ids = {sid for (j, sid) in good if j == judge_model}
        remaining = [p for p in pairs if p['id'] not in done_ids]
        
        print(f"\n{'='*60}")
        print(f"Judge: {judge_model} -- {len(done_ids)} done, {len(remaining)} remaining")
        print(f"{'='*60}")
        
        for pair in remaining:
            qid = pair['id']
            print(f"\n  {qid}: {pair['question'][:60]}...")
            
            try:
                # Pairwise x50
                pw_results = []
                errors = 0
                for i in range(repeats):
                    try:
                        j = judge_pairwise(
                            model=judge_model,
                            question=pair['question'],
                            response_a=pair['response_a']['text'],
                            response_b=pair['response_b']['text'],
                            sample_id=qid, run_index=i,
                        )
                        pw_results.append(j.to_dict())
                    except Exception as e:
                        errors += 1
                        pw_results.append({'error': str(e), 'run_index': i})
                        if errors == 1:
                            print(f"    PW err: {str(e)[:100]}")
                        if errors >= 10:
                            print(f"    Too many errors ({errors}), skipping rest of pairwise")
                            break
                    time.sleep(0.3)
                
                pw_ok = sum(1 for j in pw_results if not j.get('error'))
                print(f"    Pairwise: {pw_ok}/{repeats}")
                
                # Pointwise A x50
                pa_results = []
                errors = 0
                for i in range(repeats):
                    try:
                        j = judge_pointwise(
                            model=judge_model, question=pair['question'],
                            response_text=pair['response_a']['text'],
                            sample_id=f"{qid}_a", run_index=i,
                        )
                        pa_results.append(j.to_dict())
                    except Exception as e:
                        errors += 1
                        pa_results.append({'error': str(e), 'run_index': i})
                        if errors >= 10: break
                    time.sleep(0.3)
                
                # Pointwise B x50
                pb_results = []
                errors = 0
                for i in range(repeats):
                    try:
                        j = judge_pointwise(
                            model=judge_model, question=pair['question'],
                            response_text=pair['response_b']['text'],
                            sample_id=f"{qid}_b", run_index=i,
                        )
                        pb_results.append(j.to_dict())
                    except Exception as e:
                        errors += 1
                        pb_results.append({'error': str(e), 'run_index': i})
                        if errors >= 10: break
                    time.sleep(0.3)
                
                pa_ok = sum(1 for j in pa_results if not j.get('error'))
                pb_ok = sum(1 for j in pb_results if not j.get('error'))
                print(f"    Pointwise: A={pa_ok}/{repeats}, B={pb_ok}/{repeats}")
                
                # Compute quick metrics
                winners = [j.get('winner','?') for j in pw_results if not j.get('error')]
                if winners:
                    from collections import Counter
                    c = Counter(winners)
                    majority = c.most_common(1)[0]
                    flip = 1 - majority[1]/len(winners)
                    print(f"    Flip rate: {flip:.1%} | Majority: {majority[0]} ({majority[1]}/{len(winners)})")
                
                pa_scores = [j['score'] for j in pa_results if not j.get('error') and j.get('score') is not None]
                pb_scores = [j['score'] for j in pb_results if not j.get('error') and j.get('score') is not None]
                if pa_scores and pb_scores:
                    import numpy as np
                    try:
                        print(f"    Scores: A={np.mean(pa_scores):.1f}+/-{np.std(pa_scores):.2f} | B={np.mean(pb_scores):.1f}+/-{np.std(pb_scores):.2f}")
                    except Exception:
                        print(f"    Scores: A={len(pa_scores)} valid | B={len(pb_scores)} valid")
                
                result = {
                    'sample_id': qid,
                    'category': pair.get('category', ''),
                    'judge_model': judge_model,
                    'response_a_model': pair['response_a']['model'],
                    'response_b_model': pair['response_b']['model'],
                    'pairwise': pw_results,
                    'pointwise_a': pa_results,
                    'pointwise_b': pb_results,
                }
                all_results.append(result)
                good[(judge_model, qid)] = result
                save()
                
            except Exception as e:
                print(f"    FATAL: {e}")
                traceback.print_exc()
                save()
                continue
    
    final = Path('results') / f'exp1_v2_{ts}.json'
    with open(final, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nDone! {len(all_results)} results -> {final}")

if __name__ == '__main__':
    main()
