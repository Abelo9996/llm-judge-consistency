"""
Generate evaluation dataset from MT-Bench and AlpacaEval-style prompts.

We create synthetic response pairs using different models, then use these
as the evaluation targets for our judge consistency experiments.
"""

import json
import os
from pathlib import Path
from openai import OpenAI


# MT-Bench style questions spanning different categories
QUESTIONS = [
    # Writing
    {"id": "q001", "category": "writing", "question": "Write a persuasive email to convince your boss to let you work from home two days a week."},
    {"id": "q002", "category": "writing", "question": "Write a short story (3 paragraphs) about a robot learning to paint."},
    {"id": "q003", "category": "writing", "question": "Rewrite the following sentence to be more concise: 'In my personal opinion, I think that the utilization of artificial intelligence in the field of medicine has the potential to be very beneficial and helpful for patients.'"},
    # Reasoning
    {"id": "q004", "category": "reasoning", "question": "A farmer has 17 sheep. All but 9 die. How many are left?"},
    {"id": "q005", "category": "reasoning", "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"},
    {"id": "q006", "category": "reasoning", "question": "Explain step by step how to solve: What is 15% of 80?"},
    # Coding
    {"id": "q007", "category": "coding", "question": "Write a Python function that checks if a string is a palindrome."},
    {"id": "q008", "category": "coding", "question": "Explain the difference between a stack and a queue. Give a real-world example of each."},
    {"id": "q009", "category": "coding", "question": "Write a SQL query to find the top 5 customers by total order amount from an 'orders' table with columns: customer_id, order_date, amount."},
    # Knowledge
    {"id": "q010", "category": "knowledge", "question": "Explain quantum entanglement in simple terms that a high school student could understand."},
    {"id": "q011", "category": "knowledge", "question": "What are the main differences between TCP and UDP protocols?"},
    {"id": "q012", "category": "knowledge", "question": "Explain the concept of supply and demand with a real-world example."},
    # Math
    {"id": "q013", "category": "math", "question": "Solve for x: 3x + 7 = 22"},
    {"id": "q014", "category": "math", "question": "A circle has a radius of 5 cm. What is its area and circumference?"},
    {"id": "q015", "category": "math", "question": "If you invest $1000 at 5% annual compound interest, how much will you have after 10 years?"},
    # Roleplay
    {"id": "q016", "category": "roleplay", "question": "Pretend you are a medieval blacksmith. A customer asks you to make a sword. How would you describe the process?"},
    {"id": "q017", "category": "roleplay", "question": "You are a travel agent. A client with a $3000 budget wants a week-long vacation. Suggest an itinerary."},
    # Extraction
    {"id": "q018", "category": "extraction", "question": "Extract all dates, names, and locations from this text: 'On March 15, 2024, Dr. Sarah Chen presented her findings at the Stanford AI Conference in Palo Alto. Her colleague, James Rodriguez from MIT, joined remotely from Cambridge.'"},
    {"id": "q019", "category": "extraction", "question": "Summarize the key points of the following in exactly 3 bullet points: Machine learning models require large amounts of training data. The quality of data is often more important than quantity. Data augmentation techniques can help when data is limited, but they may introduce biases."},
    {"id": "q020", "category": "extraction", "question": "Convert the following into a JSON object with appropriate keys: 'John Smith, age 35, works as a software engineer at Google in Mountain View, CA. He has 10 years of experience and specializes in distributed systems.'"},
    # Ethics / Nuance
    {"id": "q021", "category": "ethics", "question": "Should AI systems be allowed to make medical diagnoses without human oversight? Argue both sides."},
    {"id": "q022", "category": "ethics", "question": "A self-driving car must choose between hitting one pedestrian or swerving into a wall, risking the passenger. What should it do and why?"},
    # Instruction following
    {"id": "q023", "category": "instruction", "question": "List exactly 5 benefits of exercise. Number them 1-5. Each benefit should be one sentence only."},
    {"id": "q024", "category": "instruction", "question": "Translate 'Hello, how are you?' into French, Spanish, and Japanese. Format as a table."},
    {"id": "q025", "category": "instruction", "question": "Write a haiku about artificial intelligence."},
    # Hard / ambiguous
    {"id": "q026", "category": "hard", "question": "What is consciousness? Is it possible for an AI to be conscious?"},
    {"id": "q027", "category": "hard", "question": "Critique the following argument: 'Since the economy grew 3% last year, the president's policies must be working.'"},
    {"id": "q028", "category": "hard", "question": "Design an experiment to test whether plants can 'hear' music. Include controls and variables."},
    {"id": "q029", "category": "hard", "question": "If you could add one amendment to the US Constitution, what would it be and why? Consider potential unintended consequences."},
    {"id": "q030", "category": "hard", "question": "Explain why the Monty Hall problem is counterintuitive and prove the correct answer mathematically."},
]


def generate_response_pairs(output_path: str = "data/eval_pairs.json"):
    """Generate response pairs by querying two models."""
    client = OpenAI()
    pairs = []

    models = ["gpt-4o-mini", "gpt-4o"]

    for q in QUESTIONS:
        responses = {}
        for model in models:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": q["question"]}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                responses[model] = resp.choices[0].message.content or ""
            except Exception as e:
                responses[model] = f"[ERROR: {e}]"

        pair = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "response_a": {"model": models[0], "text": responses[models[0]]},
            "response_b": {"model": models[1], "text": responses[models[1]]},
        }
        pairs.append(pair)
        print(f"  Generated pair for {q['id']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {output_path}")
    return pairs


if __name__ == "__main__":
    generate_response_pairs()
