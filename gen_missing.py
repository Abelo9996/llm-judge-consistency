"""Finish generating the last missing pair."""
import json, os
from openai import OpenAI

client = OpenAI()
with open("data/eval_pairs.json") as f:
    pairs = json.load(f)

existing_ids = {p["id"] for p in pairs}
from src.data_gen import QUESTIONS

for q in QUESTIONS:
    if q["id"] not in existing_ids:
        print(f"Generating {q['id']}...")
        responses = {}
        for model in ["gpt-4o-mini", "gpt-4o"]:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q["question"]}],
                temperature=0.7, max_tokens=1024,
            )
            responses[model] = r.choices[0].message.content or ""
            print(f"  {model} done")
        pairs.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "response_a": {"model": "gpt-4o-mini", "text": responses["gpt-4o-mini"]},
            "response_b": {"model": "gpt-4o", "text": responses["gpt-4o"]},
        })

with open("data/eval_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)
print(f"Done: {len(pairs)} pairs total")
