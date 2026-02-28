"""Quick data gen with progress output."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import json, os
from openai import OpenAI
from src.data_gen import QUESTIONS

client = OpenAI()
pairs = []
models = ["gpt-4o-mini", "gpt-4o"]

for q in QUESTIONS:
    print(f"Generating {q['id']}...", flush=True)
    responses = {}
    for model in models:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q["question"]}],
                temperature=0.7,
                max_tokens=1024,
            )
            responses[model] = r.choices[0].message.content or ""
            print(f"  {model} done", flush=True)
        except Exception as e:
            responses[model] = f"[ERROR: {e}]"
            print(f"  {model} FAILED: {e}", flush=True)

    pairs.append({
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "response_a": {"model": models[0], "text": responses[models[0]]},
        "response_b": {"model": models[1], "text": responses[models[1]]},
    })

    # Save incrementally
    os.makedirs("data", exist_ok=True)
    with open("data/eval_pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)

print(f"\nSaved {len(pairs)} pairs to data/eval_pairs.json")
