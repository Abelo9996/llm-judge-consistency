import os
try:
    from dotenv import load_dotenv
    print("dotenv installed")
except:
    print("dotenv NOT installed")

key = os.environ.get("OPENAI_API_KEY", "")
print(f"Env key before dotenv: {key[:20]}..." if key else "No OPENAI_API_KEY in env")

# Try loading
try:
    load_dotenv(override=True)
    key2 = os.environ.get("OPENAI_API_KEY", "")
    print(f"After dotenv: {key2[:20]}..." if key2 else "Still no key after dotenv")
except Exception as e:
    print(f"dotenv error: {e}")

# Manual fallback
with open(".env") as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

key3 = os.environ.get("OPENAI_API_KEY", "")
print(f"After manual: {key3[:20]}..." if key3 else "Still nothing")

# Now test
from openai import OpenAI
client = OpenAI()
r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":"hi"}], max_tokens=5)
print(f"API test: {r.choices[0].message.content}")
