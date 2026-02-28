"""Debug: check what went wrong with Paper 2 metrics on q002+."""
import json

with open('results/exp1_intra_20260225_224253_partial.json') as f:
    results = json.load(f)

for r in results[1:4]:
    sid = r['sample_id']
    pw = r['pairwise']['judgments']
    valid_pw = [j for j in pw if not j.get('error')]
    err_pw = [j for j in pw if j.get('error')]
    winners = [j['winner'] for j in valid_pw]
    print(f'\n{sid}:')
    print(f'  Pairwise: {len(valid_pw)} valid, {len(err_pw)} errored')
    print(f'  Winners: {winners[:10]}')
    if err_pw:
        print(f'  First error: {err_pw[0]["error"][:100]}')
    
    pt_a = r['pointwise_a']['judgments']
    valid_pt = [j for j in pt_a if not j.get('error')]
    scores = [j['score'] for j in valid_pt]
    print(f'  Pointwise A: {len(valid_pt)} valid, scores: {scores[:10]}')
