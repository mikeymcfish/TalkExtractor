import json

with open("labelstudio_tasks.json") as f:
    data = json.load(f)

with open("reformat_dataset.jsonl", "w") as out:
    for task in data:
        orig = task["data"]["original"].replace("\n", " ")
        ref = task["data"]["reformatted"].replace("\n", " ")
        rec = {"text": f"Original: {orig}\nReformatted:", "target": ref}
        out.write(json.dumps(rec) + "\n")
