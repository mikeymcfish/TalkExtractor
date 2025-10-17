import logging
import os, time
from typing import Optional
from openai import OpenAI

LOG = logging.getLogger("talk_extractor.teacher")

def _load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file if present."""
    if not path:
        return
    if not os.path.exists(path):
        LOG.debug(".env file not found at %s", path)
        return
    loaded_keys = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                val = value.strip().strip("'\"")
                os.environ[key] = val
                loaded_keys.append(key)
    except OSError as exc:
        LOG.warning("Unable to read .env file %s: %s", path, exc)
        return
    if loaded_keys:
        LOG.info("Loaded %s secrets from %s", len(loaded_keys), path)


_load_env_file(os.getenv("ENV_FILE", ".env"))

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
    LOG.info(
        "Calling OpenAI teacher model=%s temperature=%.2f chars=%s",
        model,
        float(temperature),
        len(passage),
    )
    for attempt in range(max_retries + 1):
        try:
            LOG.debug("Teacher attempt %s", attempt + 1)
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
            )
            out = resp.output_text
            if out and out.strip():
                LOG.debug("Teacher success on attempt %s output_chars=%s", attempt + 1, len(out))
                return out
            LOG.warning("Teacher response empty on attempt %s", attempt + 1)
        except Exception as exc:
            LOG.warning("Teacher attempt %s failed: %s", attempt + 1, exc)
            time.sleep(0.5 * (attempt + 1))
        prompt = prompt + STRICT_SUFFIX
    LOG.error("Teacher failed to return output after %s attempts", max_retries + 1)
    return None
