
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset
from ftfy import fix_text
import regex as re

DEF_CHUNK = 1200

CANDIDATE_TEXT_FIELDS = ["text", "content", "body", "article", "raw"]

LOG = logging.getLogger("talk_extractor.data_io")

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

SPEAKER_HINT = re.compile(
    r"\b([A-Z][\w']{1,})\s+(?:said|says|replied|asked|answered|shouted|cried|whispered|muttered|called|remarked|yelled|added|responded)",
    re.IGNORECASE,
)

@dataclass
class SelectionStats:
    read: int = 0
    selected: int = 0
    skipped_start: int = 0
    skipped_interval: int = 0
    estimated_speakers: float = 0.0

    def note_read(self) -> None:
        self.read += 1

    def note_skip_start(self) -> None:
        self.skipped_start += 1

    def note_skip_interval(self) -> None:
        self.skipped_interval += 1

    def note_selection(self, speaker_estimate: float) -> None:
        self.selected += 1
        self.estimated_speakers += max(speaker_estimate, 1.0)

    @property
    def skipped(self) -> int:
        return self.skipped_start + self.skipped_interval

    @property
    def average_speakers(self) -> float:
        if not self.selected:
            return 0.0
        return self.estimated_speakers / float(self.selected)

@dataclass
class SourceStats:
    raw_examples: int = 0
    candidate_passages: int = 0
    yielded_passages: int = 0
    no_text: int = 0
    below_min_words: int = 0
    insufficient_quotes: int = 0
    merged_tests: int = 0
    merged_below_min_words: int = 0
    merged_insufficient_quotes: int = 0

def estimate_speakers(passage: str) -> float:
    """Very rough heuristic of speaker variety based on quotes and attributions."""
    quote_pairs = passage.count('"') // 2
    hints = {m.group(1).strip().lower() for m in SPEAKER_HINT.finditer(passage)}
    if hints:
        return float(len(hints))
    if quote_pairs:
        # assume at least narrator + quoted speakers
        return float(min(max(2, quote_pairs), 6))
    # fallback: narrator only
    return 1.0

def iter_passages_streaming(dataset_id: str, split: str = "train", min_words: int = 80,
                            chunk: int = DEF_CHUNK, quote_pairs: int = 0,
                            stats: Optional[SourceStats] = None, pre_filter: bool = True):
    """Stream records without downloading full dataset; yields normalized, chunked passages."""
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for ex in ds:
        if stats:
            stats.raw_examples += 1
        raw = pick_text(ex) or ""
        if not raw.strip():
            if stats:
                stats.no_text += 1
            continue
        tx = ascii_quotes(fix_text(raw)).strip()
        for p in split_passages(tx, max_chars=int(chunk)):
            if stats:
                stats.candidate_passages += 1
            keep = True
            if pre_filter:
                if len(p.split()) < int(min_words):
                    keep = False
                    if stats:
                        stats.below_min_words += 1
                elif quote_pairs and not has_enough_quotes(p, min_pairs=quote_pairs):
                    keep = False
                    if stats:
                        stats.insufficient_quotes += 1
            if not keep:
                continue
            if stats:
                stats.yielded_passages += 1
            yield p

def stream_passages(src_mode: str, dataset_id: str, upload_file,
                    min_words: int, chunk: int, quote_pairs: int = 0, pre_filter: bool = True) -> Tuple[Iterable[str], str, SourceStats]:
    """Return an iterator over filtered passages plus the actual dataset identifier."""
    stats = SourceStats()
    if src_mode == "HF Dataset":
        LOG.info("Streaming HF dataset '%s' (min_words=%s, chunk=%s, quote_pairs=%s, pre_filter=%s)", dataset_id, min_words, chunk, quote_pairs, pre_filter)

        def generator():
            yield from iter_passages_streaming(
                dataset_id,
                split="train",
                min_words=min_words,
                chunk=chunk,
                quote_pairs=quote_pairs,
                stats=stats,
                pre_filter=pre_filter,
            )

        return generator(), dataset_id, stats
    if upload_file is None:
        LOG.warning("No upload file provided while upload mode selected.")
        return iter(()), "(no upload)", stats
    content = upload_file.read().decode("utf-8", errors="ignore")
    tx = ascii_quotes(fix_text(content)).strip()

    def local_passages() -> Iterable[str]:
        stats.raw_examples += 1
        if not tx:
            stats.no_text += 1
            return
        for p in split_passages(tx, max_chars=int(chunk)):
            stats.candidate_passages += 1
            keep = True
            if pre_filter:
                if len(p.split()) < int(min_words):
                    keep = False
                    stats.below_min_words += 1
                elif quote_pairs and not has_enough_quotes(p, min_pairs=quote_pairs):
                    keep = False
                    stats.insufficient_quotes += 1
            if not keep:
                continue
            stats.yielded_passages += 1
            yield p

    name = getattr(upload_file, "name", "upload.txt")
    LOG.info("Streaming upload '%s' (min_words=%s, chunk=%s, quote_pairs=%s, pre_filter=%s)", name, min_words, chunk, quote_pairs, pre_filter)
    return local_passages(), name, stats

def load_from_hub_or_upload(src_mode: str, dataset_id: str, upload_file, sample: int,
                            min_words: int, chunk: int, quote_pairs: int = 0,
                            start: int = 0, skip_every: int = 0) -> Tuple[List[str], str]:
    """Compatibility helper: consume the passage stream and return a selected list."""
    cap = int(sample) if sample else 0
    start_i = max(0, int(start)) if start is not None else 0
    skip_i = max(0, int(skip_every)) if skip_every is not None else 0
    iterator, actual_id, _ = stream_passages(src_mode, dataset_id, upload_file, min_words, chunk, quote_pairs)
    stats = SelectionStats()
    chosen: List[str] = []
    skip_gap = max(0, skip_i)
    LOG.info("Selecting passages (start=%s, skip_every=%s, cap=%s)", start_i, skip_gap, cap or "all")
    for passage in iterator:
        stats.note_read()
        if stats.read <= start_i:
            stats.note_skip_start()
            continue
        offset = stats.read - start_i - 1
        if skip_gap and (offset % (skip_gap + 1)) != 0:
            stats.note_skip_interval()
            continue
        chosen.append(passage)
        stats.note_selection(estimate_speakers(passage))
        if cap and len(chosen) >= cap:
            break
    LOG.info(
        "Completed selection: read=%s selected=%s skipped=%s avg_speakers=%.2f",
        stats.read, stats.selected, stats.skipped, stats.average_speakers,
    )
    return chosen, actual_id
