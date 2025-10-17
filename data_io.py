
from datasets import load_dataset
from ftfy import fix_text
import regex as re
from typing import List, Tuple, Iterable, Optional

DEF_CHUNK = 1200

CANDIDATE_TEXT_FIELDS = ["text", "content", "body", "article", "raw"]

def ascii_quotes(s: str) -> str:
    return (s.replace("“","\"").replace("”","\"")
            .replace("‘","'").replace("’","'")
            .replace("«","\"").replace("»","\""))

def split_passages(text: str, max_chars: int = DEF_CHUNK) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    buf, out = "", []
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip() if buf else p
        else:
            if buf: out.append(buf)
            buf = p
    if buf: out.append(buf)
    return out

def pick_text(example: dict) -> Optional[str]:
    for key in CANDIDATE_TEXT_FIELDS:
        val = example.get(key, None)
        if isinstance(val, str) and val.strip():
            return val
    # fallback: find the longest string value
    strings = [str(v) for v in example.values() if isinstance(v, str)]
    if strings:
        return max(strings, key=len)
    return None

def has_enough_quotes(passage: str, min_pairs: int = 1) -> bool:
    # Count double quotes after normalization
    q = passage.count('"')
    return (q // 2) >= min_pairs

def iter_passages_streaming(dataset_id: str, split: str = "train", min_words: int = 80, chunk: int = DEF_CHUNK, quote_pairs: int = 0):
    """Stream records without downloading full dataset; yields normalized, chunked passages."""
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for ex in ds:
        raw = pick_text(ex) or ""
        if not raw.strip():
            continue
        tx = ascii_quotes(fix_text(raw)).strip()
        for p in split_passages(tx, max_chars=int(chunk)):
            if len(p.split()) < int(min_words):
                continue
            if quote_pairs and not has_enough_quotes(p, min_pairs=quote_pairs):
                continue
            yield p

def load_from_hub_or_upload(src_mode: str, dataset_id: str, upload_file, sample: int, min_words: int, chunk: int, quote_pairs: int = 0) -> Tuple[List[str], str]:
    """Return up to `sample` passages; uses streaming for HF datasets to avoid full downloads."""
    passages: List[str] = []
    actual_id = None
    cap = int(sample) if sample else 0

    if src_mode == "HF Dataset":
        for p in iter_passages_streaming(dataset_id, split="train", min_words=min_words, chunk=chunk, quote_pairs=quote_pairs):
            passages.append(p)
            if cap and len(passages) >= cap:
                break
        actual_id = dataset_id
    else:
        if upload_file is None:
            return [], "(no upload)"
        content = upload_file.read().decode("utf-8", errors="ignore")
        tx = ascii_quotes(fix_text(content)).strip()
        for p in split_passages(tx, max_chars=int(chunk)):
            if len(p.split()) < int(min_words):
                continue
            if quote_pairs and not has_enough_quotes(p, min_pairs=quote_pairs):
                continue
            passages.append(p)
            if cap and len(passages) >= cap:
                break
        actual_id = getattr(upload_file, 'name', 'upload.txt')

    return passages, actual_id
