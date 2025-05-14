import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"

def generate_llama_response(prompt: str, max_tokens: int = 256) -> str:
    if not TOGETHER_API_KEY:
        return "❌ TOGETHER_API_KEY not set in environment."

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        start = time.time()
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        output = result["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - start
        print(f"✅ Together AI Output (in {elapsed:.2f}s):\n", output)
        return output if output else "⚠️ No output received."
    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {str(e)}"

# Run a test prompt
response = generate_llama_response("Hello, how are you?")
print("🔹 Response:\n", response)
