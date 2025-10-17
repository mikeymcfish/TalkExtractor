
import os
import gradio as gr
from typing import List, Dict, Any, Tuple

from data_io import load_from_hub_or_upload
from teacher import call_teacher, MODEL, INSTRUCTION
from validators import validate_output
from exporters import to_jsonl, to_hf_dataset

SESSION: Dict[str, Any] = {
    "passages": [],
    "records": [],
    "dataset_id": None,
}

DESCRIPTION = (
    "### Dialogue→Speaker Dataset Builder\n"
    "Prepare passages, generate `Speaker N:` dialogue via the OpenAI API, "
    "review & edit, and export JSONL / HF Datasets."
)

def on_prepare(src_mode: str, hf_id: str, upload, sample: float, min_words: float, chunk: float, quote_pairs: float) -> str:
    sample_i = int(sample) if sample else 0
    min_words_i = int(min_words) if min_words else 80
    chunk_i = int(chunk) if chunk else 1200
    qpairs_i = int(quote_pairs) if quote_pairs else 0
    passages, dataset_id = load_from_hub_or_upload(src_mode, hf_id, upload, sample_i, min_words_i, chunk_i, quote_pairs=qpairs_i)
    SESSION["passages"] = passages
    SESSION["dataset_id"] = dataset_id
    SESSION["records"] = []
    return f"Prepared {len(passages)} passages from: {dataset_id}"

def on_generate(model_name: str, temperature: float) -> Tuple[str, list]:
    if not SESSION["passages"]:
        return "No passages prepared yet.", []
    os.environ["OPENAI_MODEL"] = model_name
    rows, records, ok, bad = [], [], 0, 0
    for i, p in enumerate(SESSION["passages"]):
        y = call_teacher(p, temperature=float(temperature))
        status = "unreviewed"
        if y and validate_output(y):
            ok += 1
        else:
            bad += 1
            y = y or ""
            status = "needs_work"
        rec = {
            "task": "dialogue_format",
            "instruction": INSTRUCTION,
            "input": p,
            "output": y,
            "meta": {
                "chars": len(p),
                "model": os.getenv("OPENAI_MODEL", model_name),
                "status": status,
                "source": "LLM",
                "dataset_id": SESSION["dataset_id"]
            }
        }
        records.append(rec)
        rows.append([i, status, len(p)])
    SESSION["records"] = records
    return f"Generated {ok} valid, {bad} need work.", rows

def on_load(idx: float) -> Tuple[str, str, str]:
    i = int(idx)
    r = SESSION["records"][i]
    return r["input"], r["output"], r["meta"]["status"]

def on_save(idx: float, output: str, status: str) -> str:
    i = int(idx)
    SESSION["records"][i]["output"] = output
    SESSION["records"][i]["meta"]["status"] = status
    return f"Saved record #{i} as {status}."

def on_export_jsonl() -> str:
    path = "workspace/dataset.jsonl"
    to_jsonl(SESSION["records"], path)
    return path

def on_push(push_repo: str, private_toggle: bool) -> str:
    if not push_repo:
        return "Provide a repo name like 'yourname/gutenberg_dialogue_v1'"
    ds = to_hf_dataset(
        SESSION["records"],
        save_to="workspace/hf_dataset",
        push_repo=push_repo,
        private=bool(private_toggle),
        token=os.getenv("HF_TOKEN")
    )
    return f"Pushed {len(ds)} records to {push_repo}"

def build_ui():
    with gr.Blocks(title="Dialogue→Speaker Dataset Builder", theme=gr.themes.Default()) as demo:
        gr.Markdown("# Dialogue→Speaker Dataset Builder")
        gr.Markdown(DESCRIPTION)

        with gr.Tab("Data"):
            src_mode = gr.Radio(["HF Dataset", "Upload .txt"], value="HF Dataset", label="Source")
            hf_id = gr.Textbox(value="Navanjana/Gutenberg_books", label="HF dataset id (train split)")
            upload = gr.File(file_types=[".txt"], label="Upload a .txt file")
            sample = gr.Number(value=5, label="Sample passages (0 = all)")
            min_words = gr.Number(value=80, label="Min words per passage")
            chunk = gr.Number(value=1200, label="Chunk size (chars)")
            quote_pairs = gr.Number(value=1, label="Min dialogue quote-pairs (0 = no filter)")
            btn_prep = gr.Button("Prepare passages")
            info_data = gr.Markdown()

        with gr.Tab("Generation"):
            model_box = gr.Textbox(value=os.getenv("OPENAI_MODEL", MODEL), label="OpenAI model")
            temperature = gr.Slider(0, 1, value=0.0, step=0.1, label="Temperature")
            btn_gen = gr.Button("Generate with OpenAI")
            progress_gen = gr.Markdown()
            rec_table = gr.Dataframe(value=[], headers=["#", "status", "chars"], row_count=0, col_count=3, interactive=False)

        with gr.Tab("Review"):
            idx = gr.Number(value=0, label="Record #")
            inp = gr.Textbox(lines=12, label="Input passage", interactive=False)
            out = gr.Textbox(lines=12, label="Output (edit)")
            status = gr.Dropdown(["accepted","needs_work","unreviewed"], value="unreviewed", label="Status")
            btn_load = gr.Button("Load record")
            btn_save = gr.Button("Save changes")
            review_msg = gr.Markdown()

        with gr.Tab("Export"):
            btn_jsonl = gr.Button("Download JSONL")
            dl_path = gr.Textbox(label="JSONL path")
            push_repo = gr.Textbox(value="", label="HF Dataset repo (e.g. yourname/gutenberg_dialogue_v1)")
            private_toggle = gr.Checkbox(value=True, label="Private repo")
            btn_push = gr.Button("Push to Hugging Face Hub")
            export_msg = gr.Markdown()

        with gr.Tab("Settings"):
            instr = gr.Textbox(value=INSTRUCTION, lines=14, label="Canonical instruction (read-only)", interactive=False)
            gr.Markdown("Set `OPENAI_API_KEY` & optional `OPENAI_MODEL` in Space Secrets.")

        btn_prep.click(on_prepare, [src_mode, hf_id, upload, sample, min_words, chunk, quote_pairs], [info_data])
        btn_gen.click(on_generate, [model_box, temperature], [progress_gen, rec_table])
        btn_load.click(on_load, [idx], [inp, out, status])
        btn_save.click(on_save, [idx, out, status], [review_msg])
        btn_jsonl.click(on_export_jsonl, [], [dl_path])
        btn_push.click(on_push, [push_repo, private_toggle], [export_msg])

    return demo

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
