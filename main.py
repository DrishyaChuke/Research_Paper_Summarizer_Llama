import os
import pdfplumber
import requests
from dotenv import load_dotenv
import time

# Load TOGETHER_API_KEY from .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"
MAX_TOKENS = 512


def generate_llama_response(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
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
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        j = resp.json()
        output = j["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - start
        print(f"✅ Together AI Output (in {elapsed:.2f}s):\n{output}")
        return output or "⚠️ No output received."
    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {e}"


def reformat_text(text: str) -> str:
    system_prompt = (
        "You are a text reformatter. Fix spacing and punctuation. Do not change the content."
    )
    snippet = text[:1000]  # limit input
    prompt = f"[INST] {system_prompt}\n{snippet} [/INST]"
    try:
        out = generate_llama_response(prompt)
        print("🧾 Reformat Output:\n", out)
        return out
    except Exception as e:
        return f"❌ An error occurred while reformatting: {e}"


def summarize_text(text: str):
    system_prompt = (
        "You are a research summarizer. Summarize the content of the research paper in no more than 200 words under the headings:\n"
        "- Title and Authors\n- Objective/Problem\n- Background\n- Methods\n- Key Findings\n- Conclusion\n- Future Directions\n- Limitations\n- Potential Applications"
    )
    snippet = text[:1000]
    prompt = f"[INST] {system_prompt}\n{snippet} [/INST]"
    try:
        out = generate_llama_response(prompt, max_tokens=MAX_TOKENS)
        print("📝 Summary Output:\n", out)
        for line in out.splitlines():
            if line.strip():
                yield line.strip() + "\n"
    except Exception as e:
        yield f"❌ An error occurred while summarizing: {e}"


def extract_content_and_summarize_text(filepath):
    """Extracts text from PDF and streams reformat + summary."""
    full_text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            # try splitting columns, else full text
            lw = page.within_bbox((0, 0, page.width/2, page.height)).extract_text() or ""
            rw = page.within_bbox((page.width/2, 0, page.width, page.height)).extract_text() or ""
            ft = page.extract_text() or ""
            if len(lw) + len(rw) < 0.8 * len(ft):
                full_text += ft + "\n"
            else:
                full_text += lw + "\n" + rw + "\n"

    yield "⏳ Summarizing... please wait.\n\n"
    reformatted = reformat_text(full_text)
    for chunk in summarize_text(reformatted):
        yield chunk
