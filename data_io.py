from datasets import load_dataset
from ftfy import fix_text
import regex as re
from typing import List, Tuple

DEF_CHUNK = 1200

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

def load_from_hub_or_upload(src_mode: str, dataset_id: str, upload_file, sample: int, min_words: int, chunk: int) -> Tuple[List[str], str]:
    passages: List[str] = []
    actual_id = None
    if src_mode == "HF Dataset":
        ds = load_dataset(dataset_id, split="train")
        for ex in ds:
            raw = ex.get("text", "") or ""
            if not raw.strip():
                continue
            tx = ascii_quotes(fix_text(raw)).strip()
            for p in split_passages(tx, max_chars=int(chunk)):
                if len(p.split()) < int(min_words):
                    continue
                passages.append(p)
                if sample and len(passages) >= int(sample):
                    break
            if sample and len(passages) >= int(sample):
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
            passages.append(p)
            if sample and len(passages) >= int(sample):
                break
        actual_id = getattr(upload_file, 'name', 'upload.txt')

    return passages, actual_id
