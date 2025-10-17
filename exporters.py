import os, json
from typing import List, Dict, Any, Optional
from datasets import Dataset

def to_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def to_hf_dataset(records: List[Dict[str, Any]], save_to: Optional[str] = None,
                  push_repo: Optional[str] = None, private: bool = True, token: Optional[str] = None):
    ds = Dataset.from_list(records)
    if save_to:
        os.makedirs(save_to, exist_ok=True)
        ds.save_to_disk(save_to)
    if push_repo:
        ds.push_to_hub(push_repo, private=private, token=token)
    return ds
