
import logging
import os
import gradio as gr
from typing import Any, Dict, List, Optional, Tuple

from data_io import (
    stream_passages,
    SelectionStats,
    SourceStats,
    estimate_speakers,
    has_enough_quotes,
    scan_unique_field_values,
    inspect_dataset_fields,
)
from teacher import call_teacher, call_teacher_hf, MODEL, INSTRUCTION
from validators import validate_output
from exporters import to_jsonl, to_hf_dataset, to_labelstudio

SESSION: Dict[str, Any] = {
    "passages": [],
    "records": [],
    "dataset_id": None,
    "next_idx": 0,
    "source_stats": None,
}

DESCRIPTION = (
    "### Dialogue→Speaker Dataset Builder\n"
    "Prepare passages, generate `Speaker N:` dialogue via the OpenAI API, "
    "review & edit, and export JSONL / HF Datasets."
)

DEFAULT_LOG_LEVEL_NAME = os.getenv("TALK_EXTRACTOR_LOG_LEVEL", "INFO").upper()
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=getattr(logging, DEFAULT_LOG_LEVEL_NAME, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
LOG = logging.getLogger("talk_extractor.app")
LOG.setLevel(getattr(logging, DEFAULT_LOG_LEVEL_NAME, logging.INFO))

PREVIEW_CHARS = 160

def _make_record(passage: str, dataset_id: str) -> Dict[str, Any]:
    """Create a base record structure for later LLM completion."""
    return {
        "task": "dialogue_format",
        "instruction": INSTRUCTION,
        "input": passage,
        "output": "",
        "meta": {
            "chars": len(passage),
            "model": os.getenv("OPENAI_MODEL", MODEL),
            "status": "unreviewed",
            "source": "pending",
            "dataset_id": dataset_id,
        },
    }

def _records_table(records: Optional[List[Dict[str, Any]]] = None) -> List[List[Any]]:
    records = records if records is not None else (SESSION.get("records") or [])
    rows: List[List[Any]] = []
    for i, rec in enumerate(records):
        rows.append([
            i,
            rec.get("meta", {}).get("status", "unreviewed"),
            rec.get("meta", {}).get("chars", len(rec.get("input", ""))),
            len(rec.get("output") or ""),
        ])
    return rows

def _preview_rows(passages: List[str], start: int = 0, limit: int = 5) -> List[List[Any]]:
    if not passages:
        return []
    start_i = max(0, int(start))
    end_i = min(len(passages), start_i + max(0, int(limit)))
    rows: List[List[Any]] = []
    for idx in range(start_i, end_i):
        passage = passages[idx]
        snippet = passage[:PREVIEW_CHARS].replace("\n", " ")
        if len(passage) > PREVIEW_CHARS:
            snippet = f"{snippet}..."
        rows.append([
            idx,
            len(passage.split()),
            len(passage),
            snippet,
        ])
    return rows

def _status_markdown(selection: SelectionStats, source: Optional[SourceStats] = None) -> str:
    skipped_detail = f"{selection.skipped} (start {selection.skipped_start} / interval {selection.skipped_interval})"
    avg = f"{selection.average_speakers:.2f}" if selection.selected else "0.00"
    total_speakers = int(round(selection.estimated_speakers))
    lines = [
        "**Selection Status**",
        f"- **Records read:** {selection.read}",
        f"- **Selected:** {selection.selected}",
        f"- **Skipped:** {skipped_detail}",
        f"- **Estimated speakers (avg):** {avg} (total {total_speakers})",
    ]
    if source:
        lines.extend([
            "",
            "**Source Filtering**",
            f"- **Raw items processed:** {source.raw_examples}",
            f"- **Candidate passages:** {source.candidate_passages}",
            f"- **Yielded passages:** {source.yielded_passages}",
            f"- **Filtered (no text):** {source.no_text}",
            f"- **Filtered (below min words):** {source.below_min_words}",
            f"- **Filtered (insufficient dialogue):** {source.insufficient_quotes}",
            f"- **Merged evaluated:** {source.merged_tests}",
            f"- **Merged filtered (below min words):** {source.merged_below_min_words}",
            f"- **Merged filtered (low dialogue):** {source.merged_insufficient_quotes}",
        ])
    return "\n".join(lines)

def _load_record(idx: int) -> Tuple[str, str, str]:
    records = SESSION.get("records") or []
    if not records:
        return "", "", "unreviewed"
    clamped = max(0, min(idx, len(records) - 1))
    rec = records[clamped]
    return rec.get("input", ""), rec.get("output", ""), rec.get("meta", {}).get("status", "unreviewed")


def on_prepare(src_mode: str, hf_id: str, upload, sample: float, min_words: float, chunk: float,
               quote_pairs: float, start_record: float, skip_every: float, merge_every: float, preview_count: float,
               book_field: str, text_field: str, selected_books: list, fast_prepare: bool):
    sample_i = int(sample) if sample else 0
    min_words_i = int(min_words) if min_words else 80
    chunk_i = int(chunk) if chunk else 1200
    qpairs_i = int(quote_pairs) if quote_pairs else 0
    start_i = int(start_record) if start_record else 0
    skip_i = int(skip_every) if skip_every else 0
    preview_i = int(preview_count) if preview_count else 5
    merge_n = int(merge_every) if merge_every else 1
    merge_n = max(1, merge_n)
    cap = sample_i if sample_i > 0 else None

    LOG.info(
        "Prepare triggered src=%s dataset=%s sample=%s min_words=%s chunk=%s quotes=%s start=%s skip=%s merge=%s preview=%s",
        src_mode, hf_id, sample_i or "all", min_words_i, chunk_i, qpairs_i, start_i, skip_i, merge_n, preview_i,
    )

    progress = gr.Progress(track_tqdm=True)
    progress(0.0, desc="Initializing data preparation")
    stats = SelectionStats()
    source_stats: SourceStats = SourceStats()
    SESSION["source_stats"] = source_stats
    passages: List[str] = []
    records: List[Dict[str, Any]] = []
    pending: List[str] = []

    def snapshot(message: str):
        status_md = _status_markdown(stats, source_stats)
        preview_rows = _preview_rows(passages, start=0, limit=preview_i)
        next_idx_value = float(min(max(0, start_i), len(passages)))
        gen_preview_rows = _preview_rows(passages, start=int(next_idx_value), limit=preview_i)
        return message, status_md, preview_rows, _records_table(records), next_idx_value, gen_preview_rows

    def build_progress_message():
        return (
            f"Reading {dataset_id}: selected {stats.selected} "
            f"(read {stats.read}, skipped {stats.skipped}; no text {source_stats.no_text}, "
            f"short {source_stats.below_min_words}, low dialogue {source_stats.insufficient_quotes}, "
            f"merged total {source_stats.merged_tests}, merged short {source_stats.merged_below_min_words}, "
            f"merged low dialogue {source_stats.merged_insufficient_quotes})."
        )

    try:
        iterator, dataset_id, source_stats = stream_passages(
            src_mode,
            hf_id,
            upload,
            min_words_i,
            chunk_i,
            quote_pairs=qpairs_i,
            pre_filter=(merge_n <= 1),
            filter_field=(book_field.strip() if (src_mode == "HF Dataset" and selected_books) else None),
            allowed_values=(selected_books if (src_mode == "HF Dataset" and selected_books) else None),
            text_field=(text_field.strip() or None),
            use_fast_prepare=bool(fast_prepare),
        )
    except Exception as exc:
        LOG.exception("Failed to start passage stream: %s", exc)
        error_msg = f"Failed to load passages: {exc}"
        SESSION["passages"] = []
        SESSION["records"] = []
        SESSION["dataset_id"] = None
        SESSION["next_idx"] = 0
        SESSION["source_stats"] = source_stats
        yield snapshot(error_msg)
        return

    dataset_id = dataset_id or "(unknown)"
    book_filter_note = ""
    if src_mode == "HF Dataset" and selected_books:
        book_filter_note = f" | filtering {len(selected_books)} value(s) on '{book_field.strip() or '(field)'}'"
    text_note = f" | text field '{(text_field or '').strip() or '(auto)'}'"
    fast_note = " | fast SQL" if fast_prepare else ""
    info = f"Preparing passages from {dataset_id} (merge every {merge_n}{book_filter_note}{text_note}{fast_note})..."
    yield snapshot(info)

    skip_gap = max(0, skip_i)
    cap_reached = False

    def process_record(merged_passage: str) -> str:
        source_stats.merged_tests += 1
        words = len(merged_passage.split())
        if words < min_words_i:
            source_stats.merged_below_min_words += 1
            LOG.debug("Merged record rejected for min_words (words=%s, required=%s)", words, min_words_i)
            return "filtered_short"
        if qpairs_i and not has_enough_quotes(merged_passage, min_pairs=qpairs_i):
            source_stats.merged_insufficient_quotes += 1
            LOG.debug("Merged record rejected for insufficient dialogue (required_pairs=%s)", qpairs_i)
            return "filtered_dialogue"
        stats.note_read()
        if stats.read <= start_i:
            stats.note_skip_start()
            if stats.read == start_i and start_i:
                LOG.info("Start offset satisfied after skipping %s records", start_i)
            return "skip_start"
        offset = stats.read - start_i - 1
        if skip_gap and (offset % (skip_gap + 1)) != 0:
            stats.note_skip_interval()
            LOG.debug("Skipping record %s due to skip_every=%s", stats.read - 1, skip_gap)
            return "skip_interval"
        if cap and len(passages) >= cap:
            LOG.debug("Cap reached (%s), stopping selection.", cap)
            return "cap"
        passages.append(merged_passage)
        speaker_est = estimate_speakers(merged_passage)
        stats.note_selection(speaker_est)
        records.append(_make_record(merged_passage, dataset_id))
        LOG.debug(
            "Selected record #%s (chars=%s, speaker_est=%.2f)",
            len(passages) - 1,
            len(merged_passage),
            speaker_est,
        )
        return "selected"

    def emit_progress(message: str):
        ratio_base = cap if cap else max(1, stats.read)
        ratio = min(1.0, stats.selected / ratio_base) if ratio_base else 0.0
        progress(ratio, desc=f"Read {stats.read} | Selected {stats.selected}")
        return snapshot(message)

    for base_passage in iterator:
        pending.append(base_passage)
        if merge_n > 1 and len(pending) < merge_n:
            continue

        merged = "\n\n".join(pending)
        pending.clear()

        outcome = process_record(merged)
        if outcome == "cap":
            cap_reached = True
            break
        if outcome in ("filtered_short", "filtered_dialogue"):
            if source_stats.merged_tests <= 1 or source_stats.merged_tests % 25 == 0:
                yield emit_progress(build_progress_message())
            continue

        if outcome == "selected" or (stats.read % 25 == 0):
            yield emit_progress(build_progress_message())

        if cap and len(passages) >= cap:
            cap_reached = True
            break

    if not cap_reached and pending:
        merged = "\n\n".join(pending)
        pending.clear()
        outcome = process_record(merged)
        if outcome == "cap":
            cap_reached = True
        elif outcome in ("filtered_short", "filtered_dialogue"):
            yield emit_progress(build_progress_message())
        elif outcome == "selected" or (stats.read % 25 == 0):
            yield emit_progress(build_progress_message())

    SESSION["passages"] = passages
    SESSION["dataset_id"] = dataset_id
    SESSION["records"] = records
    SESSION["next_idx"] = min(max(0, start_i), len(passages)) if passages else 0
    SESSION["source_stats"] = source_stats
    SESSION["source_stats"] = source_stats

    final_info = (
        f"Prepared {len(passages)} passages from: {dataset_id} "
        f"(next index {SESSION['next_idx']}, skip every {skip_i}, merge every {merge_n}; read {stats.read}, skipped {stats.skipped})."
        if passages else f"No passages prepared from: {dataset_id}."
    )
    progress(1.0, desc="Preparation complete")
    LOG.info(
        "Prepare complete dataset=%s read=%s selected=%s skipped=%s avg_speakers=%.2f",
        dataset_id,
        stats.read,
        stats.selected,
        stats.skipped,
        stats.average_speakers,
    )
    LOG.info(
        "Source filtering raw=%s candidates=%s yielded=%s no_text=%s below_min=%s insufficient_quotes=%s",
        source_stats.raw_examples,
        source_stats.candidate_passages,
        source_stats.yielded_passages,
        source_stats.no_text,
        source_stats.below_min_words,
        source_stats.insufficient_quotes,
    )
    yield snapshot(final_info)

def on_generate(provider: str, model_name: str, temperature: float, start_idx: float, batch_size: float, num_batches: float) -> Tuple[str, List[List[Any]], float, List[List[Any]]]:
    if not SESSION["passages"]:
        LOG.warning("Generate requested without prepared passages.")
        return "No passages prepared yet.", [], 0.0, []
    prov = (provider or "OpenAI").strip()
    if prov == "OpenAI":
        os.environ["OPENAI_MODEL"] = model_name
    total = len(SESSION["passages"])
    start = int(start_idx) if start_idx is not None else SESSION.get("next_idx", 0)
    if start < 0:
        start = 0
    if start >= total:
        LOG.warning("Generate start index %s beyond prepared total %s", start, total)
        return f"Start index {start} is beyond the prepared passages ({total}).", _records_table(), float(total if total else 0), []
    batch = int(batch_size) if batch_size else (total - start)
    if batch <= 0:
        batch = max(1, total - start)
    batches = int(num_batches) if num_batches else 1
    if batches <= 0:
        batches = 1

    current_start = start
    overall_ok = 0
    overall_bad = 0
    batch_ranges: List[Tuple[int, int]] = []

    for batch_index in range(batches):
        if current_start >= total:
            break
        current_batch_size = batch if batch > 0 else 1
        end = min(total, current_start + current_batch_size)
        LOG.info(
            "Generate batch %s/%s start=%s end=%s provider=%s model=%s temperature=%.2f",
            batch_index + 1,
            batches,
            current_start,
            end,
            prov,
            model_name,
            float(temperature),
        )
        ok, bad = 0, 0
        for i in range(current_start, end):
            passage = SESSION["passages"][i]
            LOG.debug("Calling teacher for record %s (chars=%s)", i, len(passage))
            if prov == "HF Inference":
                y = call_teacher_hf(passage, model=model_name, temperature=float(temperature))
            else:
                y = call_teacher(passage, temperature=float(temperature))
            status = "unreviewed"
            if y and validate_output(y):
                ok += 1
            else:
                bad += 1
                y = y or ""
                status = "needs_work"
            rec = SESSION["records"][i]
            rec["output"] = y
            rec["meta"]["status"] = status
            rec["meta"]["model"] = model_name
            rec["meta"]["provider"] = prov
            rec["meta"]["source"] = "LLM"
            LOG.debug("Record %s completed status=%s output_chars=%s", i, status, len(y))
        overall_ok += ok
        overall_bad += bad
        batch_ranges.append((current_start, end - 1 if end > current_start else current_start))
        current_start = end
        if current_start >= total:
            break

    SESSION["next_idx"] = current_start if current_start < total else total
    processed = overall_ok + overall_bad
    if batch_ranges:
        first_start = batch_ranges[0][0]
        last_end = batch_ranges[-1][1]
        range_desc = f"{first_start}-{last_end}"
    else:
        range_desc = "none"
    if processed:
        progress_msg = (
            f"Generated {overall_ok} valid, {overall_bad} need work. "
            f"Processed records {range_desc} across {len(batch_ranges)} batch(es)."
        )
    else:
        progress_msg = (
            f"No records generated; start index {start} already at or beyond available passages."
        )
    LOG.info("Generate completed: %s", progress_msg)
    next_start = float(SESSION["next_idx"])
    preview_limit = batch * batches if batches else batch
    if preview_limit <= 0:
        preview_limit = batch if batch > 0 else 1
    preview_next = _preview_rows(SESSION["passages"], start=SESSION["next_idx"], limit=preview_limit)
    return progress_msg, _records_table(), next_start, preview_next

def on_load(idx: float) -> Tuple[str, str, str]:
    records = SESSION.get("records") or []
    if not records:
        LOG.warning("Load requested but no records in session.")
        return "", "", "unreviewed"
    i = max(0, min(int(idx), len(records) - 1))
    LOG.debug("Loading record %s", i)
    return _load_record(i)

def on_preview_batch(start_idx: float, batch_size: float, num_batches: float) -> List[List[Any]]:
    if not SESSION["passages"]:
        LOG.warning("Preview batch requested without prepared passages.")
        return []
    start = int(start_idx) if start_idx is not None else SESSION.get("next_idx", 0)
    if start < 0:
        start = 0
    total = len(SESSION["passages"])
    if start >= total:
        LOG.warning("Preview batch start %s beyond total %s", start, total)
        return []
    batch = int(batch_size) if batch_size else 5
    if batch <= 0:
        batch = 1
    batches = int(num_batches) if num_batches else 1
    if batches <= 0:
        batches = 1
    limit = batch * batches
    LOG.debug("Previewing batch starting at %s size %s batches=%s (limit=%s)", start, batch, batches, limit)
    return _preview_rows(SESSION["passages"], start=start, limit=limit)

def on_save(idx: float, output: str, status: str) -> str:
    records = SESSION.get("records") or []
    if not records:
        LOG.warning("Save requested without any records.")
        return "No records available to save."
    i = max(0, min(int(idx), len(records) - 1))
    records[i]["output"] = output
    records[i]["meta"]["status"] = status
    LOG.info("Record %s saved with status=%s output_chars=%s", i, status, len(output or ""))
    return f"Saved record #{i} as {status}."

def on_save_refresh(idx: float, output: str, status: str) -> Tuple[str, List[List[Any]]]:
    msg = on_save(idx, output, status)
    return msg, _records_table()

def load_record_bundle(idx: float) -> Tuple[float, str, str, str, str]:
    records = SESSION.get("records") or []
    if not records:
        LOG.warning("Load bundle requested without records.")
        return 0.0, "", "", "unreviewed", "No records loaded. Prepare or generate records first."
    i = max(0, min(int(idx) if idx is not None else 0, len(records) - 1))
    LOG.debug("Bundle loading record %s", i)
    inp, out, status = _load_record(i)
    return float(i), inp, out, status, f"Loaded record #{i}."

def step_record(idx: float, delta: int) -> Tuple[float, str, str, str, str]:
    records = SESSION.get("records") or []
    if not records:
        LOG.warning("Step requested without records.")
        return 0.0, "", "", "unreviewed", "No records loaded. Prepare or generate records first."
    base = int(idx) if idx is not None else 0
    new_idx = max(0, min(base + delta, len(records) - 1))
    LOG.debug("Stepping from %s to %s", base, new_idx)
    inp, out, status = _load_record(new_idx)
    return float(new_idx), inp, out, status, f"Loaded record #{new_idx}."

def step_prev(idx: float) -> Tuple[float, str, str, str, str]:
    return step_record(idx, -1)

def step_next(idx: float) -> Tuple[float, str, str, str, str]:
    return step_record(idx, 1)

def on_keyboard_shortcut(evt, idx: float, output: str) -> Tuple[Any, Any, Any, Any, str, List[List[Any]]]:
    records = SESSION.get("records") or []
    if not records:
        LOG.warning("Shortcut pressed with no records.")
        return 0.0, "", "", "unreviewed", "No records loaded. Prepare or generate records first.", _records_table()
    key = None
    if isinstance(evt, dict):
        key = evt.get("key")
    elif hasattr(evt, "key"):
        key = getattr(evt, "key")
    else:
        key = str(evt) if evt is not None else None
    LOG.debug("Keyboard shortcut received key=%s", key)
    actions = {
        "ArrowUp": ("accepted", 1),
        "ArrowDown": ("needs_work", 1),
        "ArrowRight": ("unreviewed", 1),
        "ArrowLeft": ("unreviewed", -1),
    }
    current_idx = max(0, min(int(idx) if idx is not None else 0, len(records) - 1))
    action = actions.get(key)
    if not action:
        LOG.debug("No action mapped for key %s", key)
        inp, out, status = _load_record(current_idx)
        return float(current_idx), inp, out, status, "", _records_table()
    status_value, delta = action
    LOG.info("Shortcut %s applies status %s and delta %s", key, status_value, delta)
    message = on_save(current_idx, output, status_value)
    new_idx = max(0, min(current_idx + delta, len(records) - 1))
    inp, out, status = _load_record(new_idx)
    if new_idx != current_idx:
        message = f"{message} Moved to record #{new_idx}."
    return float(new_idx), inp, out, status, message, _records_table()

def on_export_jsonl() -> Tuple[Any, str]:
    records = SESSION.get("records") or []
    if not records:
        LOG.warning("Export requested without records.")
        return None, "No records available to export."
    path = "workspace/dataset.jsonl"
    LOG.info("Exporting %s records to %s", len(records), path)
    to_jsonl(records, path)
    completed = sum(1 for r in records if r.get("output"))
    LOG.info("Export complete: %s records with %s outputs", len(records), completed)
    return path, f"Exported {len(records)} records ({completed} with outputs) to {path}."

def on_export_labelstudio() -> Tuple[Any, str]:
    records = SESSION.get("records") or []
    if not records:
        LOG.warning("Label Studio export requested without records.")
        return None, "No records available to export."
    path = "workspace/labelstudio_tasks.json"
    LOG.info("Exporting %s records to Label Studio JSON %s", len(records), path)
    # include model output if available; LS can show it in the UI
    to_labelstudio(records, path, include_output=True)
    completed = sum(1 for r in records if r.get("output"))
    LOG.info("Label Studio export complete: %s records (%s with outputs)", len(records), completed)
    return path, f"Exported {len(records)} records ({completed} with outputs) to {path}."

def on_push(push_repo: str, private_toggle: bool) -> str:
    if not push_repo:
        LOG.warning("Push requested without repository name.")
        return "Provide a repo name like 'yourname/gutenberg_dialogue_v1'"
    if not SESSION.get("records"):
        LOG.warning("Push requested without records.")
        return "No records available to push."
    LOG.info("Pushing %s records to HF repo %s (private=%s)", len(SESSION["records"]), push_repo, bool(private_toggle))
    ds = to_hf_dataset(
        SESSION["records"],
        save_to="workspace/hf_dataset",
        push_repo=push_repo,
        private=bool(private_toggle),
        token=os.getenv("HF_TOKEN")
    )
    LOG.info("Push complete: %s records", len(ds))
    return f"Pushed {len(ds)} records to {push_repo}"

def on_scan_books(src_mode: str, dataset_id: str, field: str, cap: float, stride: float, fast_scan: bool, current_selection: list):
    if src_mode != "HF Dataset":
        LOG.info("Scan books ignored for mode %s", src_mode)
        # no choices update; return empty update and message
        try:
            return gr.update(choices=[], value=[]), "Switch to 'HF Dataset' to scan books."
        except Exception:
            return [], "Switch to 'HF Dataset' to scan books."
    limit = int(cap) if cap else 500
    fld = (field or "").strip() or "title"
    s = int(stride) if stride else 1
    try:
        # Prefer new signature with stride
        values = scan_unique_field_values(dataset_id, fld, limit=limit, fast=bool(fast_scan), stride=s)
    except TypeError as exc:
        # Backward-compat: older function impl without stride
        LOG.info("scan_unique_field_values does not accept stride; retrying without it.")
        try:
            values = scan_unique_field_values(dataset_id, fld, limit=limit, fast=bool(fast_scan))
        except Exception as exc2:
            LOG.exception("Scan books failed (fallback): %s", exc2)
            try:
                return gr.update(choices=[], value=[]), f"Failed to scan: {exc2}"
            except Exception:
                return [], f"Failed to scan: {exc2}"
    except Exception as exc:
        LOG.exception("Scan books failed: %s", exc)
        try:
            return gr.update(choices=[], value=[]), f"Failed to scan: {exc}"
        except Exception:
            return [], f"Failed to scan: {exc}"
    msg = f"Found {len(values)} unique value(s) for '{fld}'. Select to filter during Prepare."
    # Retain selected values that are still present
    keep = []
    if isinstance(current_selection, (list, tuple)):
        keep = [v for v in current_selection if v in values]
    # Prefer returning a Gradio update (newer versions)
    try:
        return gr.update(choices=values, value=keep), msg
    except Exception:
        # Fallback for environments where update isn't available
        return values, msg

def build_ui():
    with gr.Blocks(title="Dialogue→Speaker Dataset Builder", theme=gr.themes.Default()) as demo:
        gr.Markdown("# Dialogue→Speaker Dataset Builder")
        gr.Markdown(DESCRIPTION)

        with gr.Tab("Data"):
            src_mode = gr.Radio(["HF Dataset", "Upload .txt"], value="HF Dataset", label="Source")
            hf_id = gr.Textbox(value="Navanjana/Gutenberg_books", label="HF dataset id (train split)")
            upload = gr.File(file_types=[".txt"], label="Upload a .txt file")
            with gr.Row():
                btn_load_fields = gr.Button("Load dataset fields", variant="secondary")
                book_field = gr.Dropdown(choices=[], value=None, label="Book/ID field")
                text_field = gr.Dropdown(choices=[], value=None, label="Text field to parse")
            with gr.Row():
                scan_cap = gr.Number(value=200, label="Scan up to N books", precision=0)
                scan_stride = gr.Number(value=1, label="Scan stride (every N rows)", precision=0)
                fast_scan = gr.Checkbox(value=False, label="Enable fast scan (API)")
                fast_prepare = gr.Checkbox(value=False, label="Enable fast prepare (parquet/SQL)")
                btn_scan = gr.Button("Scan books")
            book_list = gr.Dropdown(choices=[], multiselect=True, label="Books to include (optional)")
            with gr.Row():
                sample = gr.Number(value=5, label="Sample passages (0 = all)", precision=0)
                start_record = gr.Number(value=0, label="Start record #", precision=0)
                skip_every = gr.Number(value=0, label="Skip every (0 = sequential)", precision=0)
                merge_every = gr.Number(value=1, label="Merge every", precision=0)
            with gr.Row():
                min_words = gr.Number(value=80, label="Min words per passage", precision=0)
                chunk = gr.Number(value=1200, label="Chunk size (chars)", precision=0)
                quote_pairs = gr.Number(value=1, label="Min dialogue quote-pairs (0 = no filter)", precision=0)
                preview_count = gr.Number(value=5, label="Preview count", precision=0)
            with gr.Row():
                btn_prep = gr.Button("Prepare passages", variant="primary")
                btn_cancel_prep = gr.Button("Cancel Preparation", variant="stop")
            info_data = gr.Markdown()
            status_box = gr.Markdown("Status will appear here.")
            preview_table = gr.Dataframe(
                value=[],
                headers=["#", "words", "chars", "preview"],
                row_count=0,
                interactive=False,
                datatype=["number", "number", "number", "str"],
            )

        with gr.Tab("Generation"):
            provider = gr.Radio(["OpenAI", "HF Inference"], value="OpenAI", label="Provider")
            model_box = gr.Textbox(value=os.getenv("OPENAI_MODEL", MODEL), label="Model id")
            temperature = gr.Slider(0, 1, value=0.0, step=0.1, label="Temperature")
            with gr.Row():
                start_idx = gr.Number(value=0, label="Start record #", precision=0)
                batch_size = gr.Number(value=5, label="Records per batch", precision=0)
                num_batches = gr.Number(value=1, label="Number of batches", precision=0)
            with gr.Row():
                btn_preview_batch = gr.Button("Preview batch")
                btn_gen = gr.Button("Generate with OpenAI", variant="primary")
                btn_cancel_gen = gr.Button("Cancel Generation", variant="stop")
            preview_gen_table = gr.Dataframe(
                value=[],
                headers=["#", "words", "chars", "preview"],
                row_count=0,
                interactive=False,
                datatype=["number", "number", "number", "str"],
            )
            progress_gen = gr.Markdown()
            rec_table = gr.Dataframe(
                value=[],
                headers=["#", "status", "input_chars", "output_chars"],
                row_count=0,
                interactive=False,
            )

        with gr.Tab("Review"):
            gr.Markdown("Arrow shortcuts: ↑ accept, ↓ needs work, → mark & next, ← mark & previous.")
            idx = gr.Number(value=0, label="Record #", precision=0)
            status = gr.Dropdown(["unreviewed", "accepted", "needs_work"], value="unreviewed", label="Status")
            inp = gr.Textbox(lines=12, label="Input passage", interactive=False)
            out = gr.Textbox(lines=12, label="Output (edit)")
            with gr.Row():
                btn_prev = gr.Button("Previous")
                btn_load = gr.Button("Load record")
                btn_next = gr.Button("Next")
            btn_save = gr.Button("Save changes", variant="primary")
            review_msg = gr.Markdown()
            hidden_accept = gr.Button("", elem_id="shortcut-accept", visible=False)
            hidden_reject = gr.Button("", elem_id="shortcut-reject", visible=False)
            hidden_skip_prev = gr.Button("", elem_id="shortcut-skip-prev", visible=False)
            hidden_skip_next = gr.Button("", elem_id="shortcut-skip-next", visible=False)
            shortcut_up = gr.State("ArrowUp")
            shortcut_down = gr.State("ArrowDown")
            shortcut_left = gr.State("ArrowLeft")
            shortcut_right = gr.State("ArrowRight")
            gr.HTML(
                """
<script>
(() => {
  if (window.__talkExtractorHotkeysBound) {
    return;
  }
  window.__talkExtractorHotkeysBound = true;
  const selectorMap = {
    ArrowUp: '#shortcut-accept button',
    ArrowDown: '#shortcut-reject button',
    ArrowLeft: '#shortcut-skip-prev button',
    ArrowRight: '#shortcut-skip-next button',
  };
  const getRoot = () => (typeof gradioApp === 'function' ? gradioApp() : document);
  const trigger = (selector) => {
    const root = getRoot();
    if (!root) return false;
    const btn = root.querySelector(selector);
    if (!btn) return false;
    btn.click();
    return true;
  };
  window.addEventListener(
    'keydown',
    (event) => {
      const selector = selectorMap[event.key];
      if (!selector) {
        return;
      }
      const handled = trigger(selector);
      if (handled) {
        event.preventDefault();
        event.stopPropagation();
      }
    },
    true
  );
})();
</script>
                """,
                visible=True,
            )

        with gr.Tab("Export"):
            btn_jsonl = gr.Button("Create JSONL export")
            download_file = gr.File(label="Download JSONL", interactive=False, file_count="single")
            btn_ls = gr.Button("Create Label Studio JSON")
            download_ls = gr.File(label="Download Label Studio JSON", interactive=False, file_count="single")
            push_repo = gr.Textbox(value="", label="HF Dataset repo (e.g. yourname/gutenberg_dialogue_v1)")
            private_toggle = gr.Checkbox(value=True, label="Private repo")
            btn_push = gr.Button("Push to Hugging Face Hub")
            export_msg = gr.Markdown()

        with gr.Tab("Settings"):
            instr = gr.Textbox(value=INSTRUCTION, lines=14, label="Canonical instruction (read-only)", interactive=False)
            gr.Markdown("Set `OPENAI_API_KEY` & optional `OPENAI_MODEL` in Space Secrets.")
            with gr.Row():
                btn_load_fields2 = gr.Button("Load dataset fields", variant="secondary")
                # duplicate in Settings to make it accessible
                dummy = gr.Markdown("Use in Data tab to populate field dropdowns.")

        def on_load_fields(dataset_id: str):
            cols, def_id, def_text = inspect_dataset_fields(dataset_id)
            msg = (
                f"Loaded {len(cols)} columns. Default id: {def_id or '(none)'}, text: {def_text or '(none)'}"
                if cols else "Failed to load dataset fields."
            )
            try:
                return gr.update(choices=cols, value=def_id), gr.update(choices=cols, value=def_text), msg
            except Exception:
                return cols, cols, msg

        prep_event = btn_prep.click(
            on_prepare,
            [src_mode, hf_id, upload, sample, min_words, chunk, quote_pairs, start_record, skip_every, merge_every, preview_count, book_field, text_field, book_list, fast_prepare],
            [info_data, status_box, preview_table, rec_table, start_idx, preview_gen_table],
        )
        btn_scan.click(
            on_scan_books,
            [src_mode, hf_id, book_field, scan_cap, scan_stride, fast_scan, book_list],
            [book_list, info_data],
        )
        btn_load_fields.click(on_load_fields, [hf_id], [book_field, text_field, info_data])
        btn_load_fields2.click(on_load_fields, [hf_id], [book_field, text_field, info_data])
        btn_cancel_prep.click(lambda: "Preparation cancelled.", None, [info_data], cancels=[prep_event])
        btn_preview_batch.click(on_preview_batch, [start_idx, batch_size, num_batches], [preview_gen_table])
        gen_event = btn_gen.click(
            on_generate,
            [provider, model_box, temperature, start_idx, batch_size, num_batches],
            [progress_gen, rec_table, start_idx, preview_gen_table],
        )
        btn_cancel_gen.click(lambda: "Generation cancelled.", None, [progress_gen], cancels=[gen_event])
        btn_load.click(load_record_bundle, [idx], [idx, inp, out, status, review_msg])
        btn_prev.click(step_prev, [idx], [idx, inp, out, status, review_msg])
        btn_next.click(step_next, [idx], [idx, inp, out, status, review_msg])
        btn_save.click(on_save_refresh, [idx, out, status], [review_msg, rec_table])
        hidden_accept.click(on_keyboard_shortcut, [shortcut_up, idx, out], [idx, inp, out, status, review_msg, rec_table])
        hidden_reject.click(on_keyboard_shortcut, [shortcut_down, idx, out], [idx, inp, out, status, review_msg, rec_table])
        hidden_skip_prev.click(on_keyboard_shortcut, [shortcut_left, idx, out], [idx, inp, out, status, review_msg, rec_table])
        hidden_skip_next.click(on_keyboard_shortcut, [shortcut_right, idx, out], [idx, inp, out, status, review_msg, rec_table])
        btn_jsonl.click(on_export_jsonl, [], [download_file, export_msg])
        btn_ls.click(on_export_labelstudio, [], [download_ls, export_msg])
        btn_push.click(on_push, [push_repo, private_toggle], [export_msg])

    return demo

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
