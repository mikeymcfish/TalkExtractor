import os, time
from typing import Optional
from openai import OpenAI

INSTRUCTION = """You are a dialogue structuring assistant for multi-speaker TTS.

Map characters to speakers dynamically within each passage (first distinct speaker you detect -> Speaker 1, second -> Speaker 2, etc.).

Requirements:
- Detect speaker changes from context (“said/replied/asked/…”).
- Output lines strictly as:
  Speaker 1: …
  Speaker 2: …
  (and so on)
- Label narration (non-dialogue) as Speaker 1.
- Remove dialogue attribution tags (e.g., “he said”), EXCEPT when the narrator speaks in first person; keep those inline (e.g., “I said”).
- Preserve original order and content; no omissions or rewrites.
- Return only the formatted lines, no extra commentary.
"""

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()  # uses OPENAI_API_KEY

STRICT_SUFFIX = "\n\nIMPORTANT: Every line must start with 'Speaker N: ' and include at least two lines."

def call_teacher(passage: str, temperature: float = 0.0, max_retries: int = 2) -> Optional[str]:
    model = os.getenv("OPENAI_MODEL", MODEL)
    prompt = f"{INSTRUCTION}\n\nText:\n{passage}"
    for i in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
            )
            out = resp.output_text
            if out and out.strip():
                return out
        except Exception:
            time.sleep(0.5 * (i + 1))
        prompt = prompt + STRICT_SUFFIX
    return None
