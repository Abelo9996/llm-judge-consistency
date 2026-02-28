import json

d = json.load(open("results/exp1_v2_20260227_231520_partial.json"))
print(f"{len(d)} items total")
for item in d:
    sid = item["sample_id"]
    judge = item["judge_model"]
    pw = item["pairwise"]
    if isinstance(pw, list):
        ok = sum(1 for j in pw if isinstance(j, dict) and not j.get("error") and j.get("winner"))
        total = len(pw)
    elif isinstance(pw, dict) and "judgments" in pw:
        ok = sum(1 for j in pw["judgments"] if isinstance(j, dict) and not j.get("error"))
        total = len(pw["judgments"])
    else:
        ok, total = 0, 0
    print(f"  {sid} ({judge}): PW={ok}/{total}")
