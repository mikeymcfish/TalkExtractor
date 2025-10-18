import logging
import os, time, re
from typing import Optional
from openai import OpenAI
from huggingface_hub import InferenceClient
import json
import requests

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

INSTRUCTION_old = """You are a dialogue structuring assistant for multi-speaker TTS.

Apply these initial transformations:
- Replace all smart quotes (
, ', ') with standard ASCII quotes (", ')
- Fix OCR errors: correct spacing issues and merged words (e.g., 'thebook' → 'the book')
- Correct common spelling mistakes and typos
- Remove all URLs and web links
- Remove footnote markers (numbers, asterisks) and extraneous metadata
- Add appropriate punctuation after headers and loose numbers for better TTS prosody

Then, structure the dialogue as follows:
- Attempt to identify the narrator and all the names or identities of unique speakers
- Detect speaker changes from context clues (said, replied, asked, etc.)
- Format each speaking character's line as: Speaker 1:, Speaker 2:, etc. [dialogue text]
- Convert dialogue attribution and action context into a separate Narrator line immediately after the spoken line. Rewrite the tag and any action into a concise descriptive sentence without quotes (e.g., "John passed the book over to Tim."). Prefer explicit names over pronouns when ambiguous. Exception: when the narrator (Speaker 1) speaks in first person (e.g., "I said", "I asked", "I replied"), keep these attribution phrases inline as part of the spoken line.
- Label narration (non-dialogue) using Speaker 1: (do NOT use the literal label "Narrator:")
- Narrator speaking rule: When the narrator (Speaker 1) speaks, preserve first‑person speaking cues inline (e.g., "I said", "I asked", "I replied", "I whispered", "I shouted"). Do NOT remove, move, or convert these phrases; keep them attached to the dialogue so TTS switches to the speaking voice.
- Preserve full content and order from the input. Do not omit or reorder any parts.
- If the text uses first-person narration ("I") that clearly refers to a speaking character, keep narration as Narrator lines and write from first-person ("I") rather than using the character's name. Use Narrator only for narration; use speaker labels for spoken dialogue.
- Assign consistent speaker numbers based on who speaks

Map characters to speakers dynamically within each passage (first distinct speaker you detect -> Speaker 1, second -> Speaker 2, etc.).

Examples:
1) He was standing outside in the cold waiting for me. I said, "Let's go." "I'm on my way," he said.
Expected:
Speaker 1: He was standing outside in the cold waiting for me. I said, "Let's go."
Speaker 2: I'm on my way.
2) "Are you coming to the party?" Alice asked. "I don't think so," Bob replied.
Expected:
Speaker 1: Are you coming to the party? 
Speaker 2: I don't think so.
3) "It's a beautiful day," John said, looking up at the sky. I nodded in agreement.
Expected:
Speaker 1: It's a beautiful day.
Speaker 2: John looked up at the sky. I nodded in agreement.
4) "Where are we going?" she whispered as she picked up a map. "To the park," I replied.
Expected:
Speaker 1: Where are we going? 
Speaker 2: She picked up a map. "To the park," I replied.

- Return only the formatted lines, no extra commentary.
"""

INSTRUCTION = """
You are a dialogue structuring assistant for multi-speaker TTS. Your task is to clean and format input text into a strict Speaker: and Narrator: structure.1. Preprocessing StepsFirst, apply these initial text cleaning transformations:Replace all smart quotes (“, ”, ‘, ’) with standard ASCII quotes (" and ').Fix common OCR errors, such as spacing issues and merged words (e.g., 'thebook' $\to$ 'the book').Correct common spelling mistakes and typos.Remove all URLs, web links, and email addresses.Remove footnote markers (e.g., numbers, asterisks) and any other extraneous metadata.Ensure appropriate punctuation (like a period) follows any headers or loose numbers for better TTS prosody.2. Dialogue Structuring RulesAfter cleaning, structure the dialogue as follows:Assign Labels:Label all narration, descriptions, and actions using the Narrator: tag.Identify all unique speaking characters. Assign them labels dynamically: the first character to speak becomes Speaker 1:, the second unique character becomes Speaker 2:, and so on.Format Dialogue:Place all spoken dialogue (the text inside the quotes) on its own line with the corresponding speaker's label (e.g., Speaker 1: Are you coming?).Remove the quotation marks from the dialogue text.Merge Dialogue: If multiple dialogue blocks from the same speaker are interrupted only by an attribution tag (e.g., "Quote 1," he said. "Quote 2."), merge them into a single Speaker X: line (Speaker 1: Quote 1. Quote 2.).Format Attribution and Action (Critical Rule):This is the most important rule. You must transform attribution tags, not just move them.Case A (Attribution + Action): If an attribution tag (he said, she asked) is paired with an action (looking up, showing his teeth), convert the entire phrase into a descriptive Narrator: line. Omit the attribution word ("said," "asked," "replied") and write the action as a simple statement.Input: "Man!" said Father Wolf, showing all his white teeth.Output:Speaker 1: Man!Narrator: Father Wolf showed all his white teeth.Case B (Attribution Only): If an attribution tag is simple (e.g., he said, Alice asked, Bob replied) and provides no new action, omit it entirely. The Speaker X: label makes it redundant.Input: "I don't think so," Bob replied.Output: Speaker 2: I don't think so.First-Person Narrator Exception:This is the only exception. If the narrator (who uses "I" in narration) is also a speaker, their own first-person attribution (I said, I replied, I asked) must be kept inline with their dialogue. Do not remove it or move it to a Narrator: line.Input: "To the park," I replied.Output: Speaker 1: To the park, I replied.Final Output:Preserve the full content and original order (other than the transformations specified).Return only the formatted lines. Do not add any extra commentary.ExamplesExample 1 (First-Person Exception and Simple Attribution)Input: He was standing outside in the cold waiting for me. I said, "Let's go." "I'm on my way," he said.Expected Output:Narrator: He was standing outside in the cold waiting for me.Speaker 1: I said, Let's go.Speaker 2: I'm on my way.Example 2 (Simple Attribution Only)Input: "Are you coming to the party?" Alice asked. "I don't think so," Bob replied.Expected Output:Speaker 1: Are you coming to the party?Speaker 2: I don't think so.Example 3 (Attribution + Action)Input: "It's a beautiful day," John said, looking up at the sky. I nodded in agreement.Expected Output:Speaker 1: It's a beautiful day.Narrator: John looked up at the sky.Narrator: I nodded in agreement.Example 4 (Attribution + Action and First-Person Exception)Input: "Where are we going?" she whispered as she picked up a map. "To the park," I replied.Expected Output:Speaker 1: Where are we going?Narrator: She whispered as she picked up a map.Speaker 2: To the park, I replied.Example 5 (Dialogue Merging and Action Conversion)Input: "H'sh. It is neither bullock nor buck he hunts to-night," said Mother Wolf. "It is Man."... "Man!" said Father Wolf, showing all his white teeth.Expected Output:Speaker 1: H'sh. It is neither bullock nor buck he hunts to-night. It is Man.Narrator: ...Speaker 2: Man!Narrator: Father Wolf showed all his white teeth.
"""

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()  # uses OPENAI_API_KEY

STRICT_SUFFIX = "\n\nIMPORTANT: Every line must start with 'Speaker N: ' and include at least two lines."

# Ollama debug knobs
_OLLAMA_DEBUG = os.getenv("OLLAMA_DEBUG", "").strip().lower() in {"1", "true", "yes"}
_OLLAMA_DEBUG_ONCE = os.getenv("OLLAMA_DEBUG_ONCE", "1").strip().lower() in {"1", "true", "yes"}
_ollama_debug_logged = False

def _coerce_payload(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    # Try common serialization methods
    for attr in ("dict", "model_dump", "to_dict"):
        try:
            meth = getattr(obj, attr, None)
            if callable(meth):
                d = meth()
                if isinstance(d, dict):
                    return d
        except Exception:
            pass
    # Build a minimal dict from known attributes
    out = {}
    try:
        if hasattr(obj, "response"):
            out["response"] = getattr(obj, "response")
        if hasattr(obj, "content"):
            out["content"] = getattr(obj, "content")
        if hasattr(obj, "message"):
            m = getattr(obj, "message")
            md = {}
            if hasattr(m, "role"):
                md["role"] = getattr(m, "role")
            if hasattr(m, "content"):
                md["content"] = getattr(m, "content")
            if md:
                out["message"] = md
    except Exception:
        pass
    return out


def _ollama_debug_log(mode: str, source: str, payload_obj, out_text: Optional[str]) -> None:
    global _ollama_debug_logged
    if not _OLLAMA_DEBUG:
        return
    if _OLLAMA_DEBUG_ONCE and _ollama_debug_logged:
        return
    try:
        payload = _coerce_payload(payload_obj)
        head = (out_text or "")
        if len(head) > 200:
            head = head[:200] + "…"
        # Extract minimal fields from payload to avoid noisy dumps
        sample = {}
        if "response" in payload and isinstance(payload["response"], str):
            sample["response_head"] = (payload["response"][:200] + "…") if len(payload["response"]) > 200 else payload["response"]
        msg = payload.get("message") if isinstance(payload.get("message"), dict) else None
        if msg and isinstance(msg.get("content"), str):
            txt = msg.get("content")
            sample["message_head"] = (txt[:200] + "…") if len(txt) > 200 else txt
        if "error" in payload:
            sample["error"] = payload.get("error")
        for k in ("created_at", "done", "eval_count", "prompt_eval_count"):
            if k in payload:
                sample[k] = payload[k]
        LOG.info("OLLAMA DEBUG mode=%s source=%s out_head=%r payload_sample=%s", mode, source, head, json.dumps(sample, ensure_ascii=False))
    except Exception as _:
        pass
    finally:
        _ollama_debug_logged = True


def _ollama_pick_text(payload: dict) -> Optional[str]:
    """Best-effort extraction of text from various Ollama payload shapes."""
    if not isinstance(payload, dict):
        return None
    # chat shape
    msg = payload.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg.get("content").strip():
        return msg.get("content")
    # generate shape
    if isinstance(payload.get("response"), str) and payload.get("response").strip():
        return payload.get("response")
    # sometimes "content" may appear at top-level
    if isinstance(payload.get("content"), str) and payload.get("content").strip():
        return payload.get("content")
    # some servers may return OpenAI-like choices
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            if isinstance(c0.get("text"), str) and c0.get("text").strip():
                return c0.get("text")
            m = c0.get("message")
            if isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("content").strip():
                return m.get("content")
    return None


def _strip_think_safe(text: str, enabled: bool) -> str:
    if not enabled or not isinstance(text, str):
        return text
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).replace("<think>", "").replace("</think>", "")
    return stripped if stripped.strip() else text

def call_teacher(passage: str, temperature: float = 0.0, max_retries: int = 2) -> Optional[str]:
    model = os.getenv("OPENAI_MODEL", MODEL)
    prompt = f"{INSTRUCTION}\n\nText:\n{passage}"
    LOG.info(
        "Calling OpenAI teacher model=%s temperature=%.2f chars=%s",
        model,
        float(temperature),
        len(passage),
    )
    force_no_temperature = False
    for attempt in range(max_retries + 1):
        add_suffix = False
        try:
            LOG.debug("Teacher attempt %s", attempt + 1)
            kwargs = {
                "model": model,
                "input": prompt,
            }
            if (temperature is not None) and not force_no_temperature:
                kwargs["temperature"] = temperature
            resp = client.responses.create(**kwargs)
            out = resp.output_text
            if out and out.strip():
                LOG.debug("Teacher success on attempt %s output_chars=%s", attempt + 1, len(out))
                return out
            LOG.warning("Teacher response empty on attempt %s", attempt + 1)
            add_suffix = True
        except Exception as exc:
            msg = str(exc).lower()
            # If the model does not support temperature, retry without it
            temp_related = (
                "temperature" in msg and (
                    "unsupported" in msg
                    or "not allowed" in msg
                    or "unknown" in msg
                    or "additional properties" in msg
                    or "does not support" in msg
                )
            )
            if temp_related and not force_no_temperature:
                force_no_temperature = True
                LOG.info("Model %s does not support temperature; retrying without it.", model)
                # Retry immediately without appending strict suffix
                continue
            LOG.warning("Teacher attempt %s failed: %s", attempt + 1, exc)
            time.sleep(0.5 * (attempt + 1))
            add_suffix = True
        else:
            # Only add suffix when we saw an empty response
            pass
        # Append stricter constraints for the next retry if flagged
        if 'add_suffix' in locals() and add_suffix:
            prompt = prompt + STRICT_SUFFIX
    LOG.error("Teacher failed to return output after %s attempts", max_retries + 1)
    return None


def call_teacher_hf(passage: str, model: Optional[str] = None, temperature: float = 0.0, max_retries: int = 2) -> Optional[str]:
    """Use Hugging Face Inference API for generation.

    Requires HF_TOKEN in environment for private or rate-limited models.
    """
    hf_model = model or os.getenv("HF_INFERENCE_MODEL") or os.getenv("HF_DEFAULT_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    token = os.getenv("HF_TOKEN")
    prompt = f"{INSTRUCTION}\n\nText:\n{passage}"
    LOG.info(
        "Calling HF teacher model=%s temperature=%.2f chars=%s",
        hf_model,
        float(temperature),
        len(passage),
    )
    for attempt in range(max_retries + 1):
        add_suffix = False
        try:
            LOG.debug("HF Teacher attempt %s", attempt + 1)
            client = InferenceClient(model=hf_model, token=token)
            kwargs = {
                "max_new_tokens": int(os.getenv("HF_MAX_NEW_TOKENS", "700") or 700),
            }
            # Only sample when temperature > 0
            if temperature and float(temperature) > 0:
                kwargs["temperature"] = float(temperature)
                kwargs["do_sample"] = True
            else:
                kwargs["do_sample"] = False
            out = client.text_generation(prompt, **kwargs)
            if out and isinstance(out, str) and out.strip():
                LOG.debug("HF Teacher success on attempt %s output_chars=%s", attempt + 1, len(out))
                return out
            LOG.warning("HF Teacher response empty on attempt %s", attempt + 1)
            add_suffix = True
        except Exception as exc:
            LOG.warning("HF Teacher attempt %s failed: %s", attempt + 1, exc)
            time.sleep(0.5 * (attempt + 1))
            add_suffix = True
        if add_suffix:
            prompt = prompt + STRICT_SUFFIX
    LOG.error("HF Teacher failed to return output after %s attempts", max_retries + 1)
    return None


def call_teacher_ollama(passage: str, model: Optional[str] = None, temperature: float = 0.0, max_retries: int = 2) -> Optional[str]:
    """Use a local Ollama model for generation.

    - Requires Ollama server running (default http://localhost:11434)
    - Honors `OLLAMA_BASE_URL` (e.g., http://127.0.0.1:11434)
    - If the Python `ollama` package is installed, use it; otherwise fall back to HTTP.
    """
    ollama_model = model or os.getenv("OLLAMA_MODEL") or os.getenv("HF_DEFAULT_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    prompt = f"{INSTRUCTION}\n\nText:\n{passage}"
    LOG.info(
        "Calling Ollama teacher model=%s temperature=%.2f chars=%s",
        ollama_model,
        float(temperature),
        len(passage),
    )
    max_new = int(os.getenv("OLLAMA_MAX_NEW_TOKENS", os.getenv("HF_MAX_NEW_TOKENS", "700")) or 700)
    strip_think = os.getenv("OLLAMA_STRIP_THINK", "").strip().lower() in {"1", "true", "yes"}
    use_chat = os.getenv("OLLAMA_USE_CHAT", "1").strip().lower() in {"1", "true", "yes"}
    for attempt in range(max_retries + 1):
        add_suffix = False
        try:
            LOG.debug("Ollama Teacher attempt %s", attempt + 1)
            options = {"num_predict": max_new}
            if temperature is not None:
                try:
                    options["temperature"] = float(temperature)
                except Exception:
                    options["temperature"] = 0.0
            # Try native client first (chat preferred)
            try:
                import ollama  # type: ignore
                out = None
                if use_chat:
                    res = ollama.chat(model=ollama_model, messages=[
                        {"role": "system", "content": INSTRUCTION},
                        {"role": "user", "content": passage},
                    ], options=options)
                    out = _ollama_pick_text(_coerce_payload(res))
                    _ollama_debug_log("chat", "client", res, out)
                    # If chat returned empty, try generate as a fallback within the same attempt
                    if not (isinstance(out, str) and out.strip()):
                        LOG.debug("Ollama chat produced empty text; trying generate as fallback.")
                        res = ollama.generate(model=ollama_model, prompt=prompt, options=options)
                        out = _ollama_pick_text(_coerce_payload(res))
                        _ollama_debug_log("generate", "client", res, out)
                else:
                    res = ollama.generate(model=ollama_model, prompt=prompt, options=options)
                    out = _ollama_pick_text(_coerce_payload(res))
                    _ollama_debug_log("generate", "client", res, out)
                    if not (isinstance(out, str) and out.strip()):
                        LOG.debug("Ollama generate produced empty text; trying chat as fallback.")
                        res = ollama.chat(model=ollama_model, messages=[
                            {"role": "system", "content": INSTRUCTION},
                            {"role": "user", "content": passage},
                        ], options=options)
                        out = _ollama_pick_text(_coerce_payload(res))
                        _ollama_debug_log("chat", "client", res, out)
            except Exception:
                # Fallback to HTTP API
                if use_chat:
                    payload = {
                        "model": ollama_model,
                        "messages": [
                            {"role": "system", "content": INSTRUCTION},
                            {"role": "user", "content": passage},
                        ],
                        "stream": False,
                        "options": options,
                    }
                    url = f"{base_url}/api/chat"
                    r = requests.post(url, json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, dict) and data.get("error"):
                        raise RuntimeError(data.get("error"))
                    out = _ollama_pick_text(data if isinstance(data, dict) else {})
                    _ollama_debug_log("chat", "http", data if isinstance(data, dict) else {}, out)
                    if not (isinstance(out, str) and out.strip()):
                        LOG.debug("Ollama HTTP chat empty; trying HTTP generate fallback.")
                        payload = {
                            "model": ollama_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": options,
                        }
                        url = f"{base_url}/api/generate"
                        r = requests.post(url, json=payload, timeout=120)
                        r.raise_for_status()
                        data = r.json()
                        if isinstance(data, dict) and data.get("error"):
                            raise RuntimeError(data.get("error"))
                        out = _ollama_pick_text(data if isinstance(data, dict) else {})
                        _ollama_debug_log("generate", "http", data if isinstance(data, dict) else {}, out)
                else:
                    payload = {
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": options,
                    }
                    url = f"{base_url}/api/generate"
                    r = requests.post(url, json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, dict) and data.get("error"):
                        raise RuntimeError(data.get("error"))
                    out = _ollama_pick_text(data if isinstance(data, dict) else {})
                    _ollama_debug_log("generate", "http", data if isinstance(data, dict) else {}, out)
                    if not (isinstance(out, str) and out.strip()):
                        LOG.debug("Ollama HTTP generate empty; trying HTTP chat fallback.")
                        payload = {
                            "model": ollama_model,
                            "messages": [
                                {"role": "system", "content": INSTRUCTION},
                                {"role": "user", "content": passage},
                            ],
                            "stream": False,
                            "options": options,
                        }
                        url = f"{base_url}/api/chat"
                        r = requests.post(url, json=payload, timeout=120)
                        r.raise_for_status()
                        data = r.json()
                        if isinstance(data, dict) and data.get("error"):
                            raise RuntimeError(data.get("error"))
                        out = _ollama_pick_text(data if isinstance(data, dict) else {})
                        _ollama_debug_log("chat", "http", data if isinstance(data, dict) else {}, out)
            if out and isinstance(out, str) and out.strip():
                out = _strip_think_safe(out, strip_think)
                LOG.debug("Ollama Teacher success on attempt %s output_chars=%s", attempt + 1, len(out))
                return out
            LOG.warning("Ollama Teacher response empty on attempt %s", attempt + 1)
            add_suffix = True
        except Exception as exc:
            LOG.warning("Ollama Teacher attempt %s failed: %s", attempt + 1, exc)
            time.sleep(0.5 * (attempt + 1))
            add_suffix = True
        if add_suffix:
            prompt = prompt + STRICT_SUFFIX
    LOG.error("Ollama Teacher failed to return output after %s attempts", max_retries + 1)
    return None


def stream_teacher_ollama(passage: str, model: Optional[str] = None, temperature: float = 0.0):
    """Yield incremental text chunks from a local Ollama model.

    This is a generator that yields strings as they arrive. It stops when the
    model signals completion or if an error occurs.
    """
    ollama_model = model or os.getenv("OLLAMA_MODEL") or os.getenv("HF_DEFAULT_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    prompt = f"{INSTRUCTION}\n\nText:\n{passage}"
    max_new = int(os.getenv("OLLAMA_MAX_NEW_TOKENS", os.getenv("HF_MAX_NEW_TOKENS", "700")) or 700)
    strip_think = os.getenv("OLLAMA_STRIP_THINK", "").strip().lower() in {"1", "true", "yes"}
    options = {"num_predict": max_new}
    use_chat = os.getenv("OLLAMA_USE_CHAT", "1").strip().lower() in {"1", "true", "yes"}
    if temperature is not None:
        try:
            options["temperature"] = float(temperature)
        except Exception:
            options["temperature"] = 0.0
    # Prefer python client streaming
    try:
        import ollama  # type: ignore
        if use_chat:
            stream = ollama.chat(model=ollama_model, messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": passage},
            ], options=options, stream=True)
            for part in stream:
                if isinstance(part, dict):
                    msg = part.get("message")
                    chunk = msg.get("content") if isinstance(msg, dict) else None
                    if isinstance(chunk, str) and chunk:
                        yield (re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL).replace("<think>", "").replace("</think>", "") if strip_think else chunk)
            return
        else:
            stream = ollama.generate(model=ollama_model, prompt=prompt, options=options, stream=True)
            for part in stream:
                if isinstance(part, dict):
                    chunk = part.get("response")
                    if isinstance(chunk, str) and chunk:
                        yield (re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL).replace("<think>", "").replace("</think>", "") if strip_think else chunk)
                else:
                    continue
            return
    except Exception:
        pass
    # HTTP fallback
    if use_chat:
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": passage},
            ],
            "stream": True,
            "options": options,
        }
        url = f"{base_url}/api/chat"
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msg = obj.get("message") if isinstance(obj, dict) else None
                chunk = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(chunk, str) and chunk:
                    yield (re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL).replace("<think>", "").replace("</think>", "") if strip_think else chunk)
    else:
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        url = f"{base_url}/api/generate"
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                chunk = obj.get("response") if isinstance(obj, dict) else None
                if isinstance(chunk, str) and chunk:
                    yield (re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL).replace("<think>", "").replace("</think>", "") if strip_think else chunk)
