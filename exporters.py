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

def to_labelstudio(records: List[Dict[str, Any]], path: str, include_output: bool = True) -> None:
    """Export tasks in Label Studio import format.

    Produces a JSON array where each item is a task with a `data` dict.
    Keys used:
      - passage: original input passage
      - generated (optional): model output, if present and include_output is True
      - meta: original metadata attached to the record (status, model, etc.)

    You can configure your Label Studio project to reference these fields
    in the labeling interface, e.g. showing both passage and generated text.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tasks: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        passage = r.get("input", "")
        data: Dict[str, Any] = {
            # Many Label Studio templates expect a field named "original".
            "original": passage,
            # Keep a friendly alias too in case you reference $passage in your project.
            "passage": passage,
        }
        if include_output and r.get("output"):
            out = r.get("output")
            # Common names for pre-existing model output
            data["generated"] = out
            data["output"] = out
        meta = r.get("meta") or {}
        if meta:
            data["meta"] = meta
        tasks.append({
            "id": i,
            "data": data,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
