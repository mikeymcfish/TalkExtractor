
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Set

from datasets import load_dataset
from ftfy import fix_text
import os
from urllib.parse import quote as urlquote
import contextlib
import regex as re
import concurrent.futures as _cf
import requests

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

def _hf_parquet_urls(dataset_id: str) -> List[str]:
    """Fetch parquet file URLs for a dataset via the public datasets-server API.

    Returns an empty list if not available.
    """
    try:
        import requests  # type: ignore
    except Exception:
        return []

def _hf_splits(dataset_id: str):
    """Return list of split descriptors using datasets-server.

    Each item is a dict with keys like 'config', 'split', 'num_examples'.
    """
    base = os.getenv("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co")
    url = f"{base}/splits?dataset={urlquote(dataset_id, safe='')}"
    headers = {}
    tok = os.getenv("HF_TOKEN", "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("splits", []) or []
    except Exception:
        return []

def _hf_rows_sample_distinct(dataset_id: str, field: str, *, limit: int,
                             page_size: int = 200, max_requests: int = 20,
                             prefer_split: str = "train") -> Optional[List[str]]:
    """Sample distinct values for a column via datasets-server /rows across the split.

    Spreads requests across the dataset to increase coverage; respects HF_TOKEN.
    """
    splits = _hf_splits(dataset_id)
    if not splits:
        return None
    chosen = None
    best = -1
    for s in splits:
        if not isinstance(s, dict):
            continue
        sp = s.get("split") or s.get("name")
        cfg = s.get("config")
        n = int(s.get("num_examples") or 0)
        if isinstance(sp, str) and sp.lower() == prefer_split:
            chosen = (cfg, sp, n)
            break
        if n > best:
            best = n
            chosen = (cfg, sp, n)
    if not chosen:
        return None
    config, split, total = chosen
    if not config or not split or not total:
        return None
    base = os.getenv("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co")
    headers = {}
    tok = os.getenv("HF_TOKEN", "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    reqs = max(1, int(max_requests))
    ps = max(1, int(page_size))
    step = max(1, total // reqs)
    offsets = [min(total - 1, i * step) for i in range(reqs)]
    seen: Set[str] = set()
    out: List[str] = []
    for off in offsets:
        url = (
            f"{base}/rows?dataset={urlquote(dataset_id, safe='')}"
            f"&config={urlquote(str(config), safe='')}"
            f"&split={urlquote(str(split), safe='')}"
            f"&offset={int(off)}&length={ps}"
        )
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code != 200:
                continue
            data = r.json()
            rows = data.get("rows") or []
            for item in rows:
                row = item.get("row") if isinstance(item, dict) else None
                if not isinstance(row, dict):
                    continue
                raw = row.get(field)
                if raw is None:
                    continue
                if isinstance(raw, (list, tuple)):
                    cand = [str(v).strip() for v in raw if v is not None]
                else:
                    cand = [str(raw).strip()]
                for v in cand:
                    if not v:
                        continue
                    key = v.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(v)
                    if len(out) >= limit:
                        return out
        except Exception:
            continue
    return out or None
    base = os.getenv("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co")
    url = f"{base}/parquet?dataset={urlquote(dataset_id, safe='')}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        files = data.get("parquet_files") or []
        urls = [f.get("url") for f in files if isinstance(f, dict) and f.get("url")]
        return [u for u in urls if isinstance(u, str)]
    except Exception:
        return []

def _sql_list_literal(urls: List[str]) -> str:
    return "[" + ", ".join("'" + u.replace("'", "''") + "'" for u in urls) + "]"

def _distinct_via_duckdb(parquet_urls: List[str], field: str, limit: int, search: Optional[str] = None) -> Optional[List[str]]:
    """Attempt fast DISTINCT scan using DuckDB over remote parquet URLs, optional case-insensitive contains filter."""
    if not parquet_urls:
        return None
    try:
        import duckdb  # type: ignore
    except Exception:
        return None
    # sanitize identifier
    ident = field.strip()
    if not ident:
        return None
    # DuckDB HTTPFS setup and conservative threading/retry to avoid 429s
    con = duckdb.connect(database=':memory:')
    with contextlib.suppress(Exception):
        con.execute("INSTALL httpfs;")
    with contextlib.suppress(Exception):
        con.execute("LOAD httpfs;")
    # Optionally set auth header to reduce rate limiting if HF_TOKEN provided
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        with contextlib.suppress(Exception):
            header = ("Authorization: Bearer " + token).replace("'", "''")
            con.execute(f"SET http_headers='{header}';")
    with contextlib.suppress(Exception):
        con.execute("PRAGMA threads=1;")
    with contextlib.suppress(Exception):
        con.execute("SET http_retries=8;")
    with contextlib.suppress(Exception):
        con.execute("SET http_retry_backoff=2.0;")
    # Use read_parquet with an inlined list of URLs (parameters not allowed for table functions)
    try:
        # Build SQL; quote identifier
        col = '"' + ident.replace('"', '""') + '"'
        list_sql = _sql_list_literal(parquet_urls)
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet({list_sql});")
        where = [f"{col} IS NOT NULL"]
        if search:
            pat = "%" + str(search).strip().lower() + "%"
            pat_sql = "'" + pat.replace("'", "''") + "'"
            where.append(f"lower(CAST({col} AS VARCHAR)) LIKE {pat_sql}")
        where_sql = " WHERE " + " AND ".join(where) if where else ""
        q = f"SELECT DISTINCT {col} AS val FROM v{where_sql} LIMIT {int(limit)}"
        res = con.execute(q).fetchall()
        values = [str(r[0]) for r in res if r and r[0] is not None]
        return sorted(values, key=lambda s: s.lower())
    except Exception:
        return None

def iter_passages_duckdb(dataset_id: str, parquet_urls: List[str], filter_field: str, allowed_values: Set[str],
                         min_words: int, chunk: int, quote_pairs: int,
                         stats: Optional[SourceStats], pre_filter: bool,
                         text_field: Optional[str] = None):
    """Yield passages using DuckDB over remote parquet, filtering by allowed book IDs.

    Only rows whose `filter_field` (lowercased string) is in `allowed_values` are scanned.
    Reads only present candidate text columns to minimize IO.
    """
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        LOG.info("DuckDB not available, cannot use fast prepare: %s", exc)
        return
    if not parquet_urls:
        return
    con = duckdb.connect(database=':memory:')
    with contextlib.suppress(Exception):
        con.execute("INSTALL httpfs;")
    with contextlib.suppress(Exception):
        con.execute("LOAD httpfs;")
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        with contextlib.suppress(Exception):
            header = ("Authorization: Bearer " + token).replace("'", "''")
            con.execute(f"SET http_headers='{header}';")
    with contextlib.suppress(Exception):
        con.execute("PRAGMA threads=1;")
    with contextlib.suppress(Exception):
        con.execute("SET http_retries=8;")
    with contextlib.suppress(Exception):
        con.execute("SET http_retry_backoff=2.0;")
    # Create view over parquet files
    list_sql = _sql_list_literal(parquet_urls)
    try:
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet({list_sql});")
    except Exception as exc:
        LOG.info("Failed to open parquet via DuckDB (will fall back): %s", exc)
        return
    # Resolve available columns and which text fields exist
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info('v');").fetchall()]
    except Exception as exc:
        LOG.info("Failed to inspect parquet schema: %s", exc)
        return
    # Decide text column(s)
    candidate_text: List[str]
    if text_field and (text_field in cols):
        candidate_text = [text_field]
    else:
        candidate_text = [c for c in CANDIDATE_TEXT_FIELDS if c in cols]
        if not candidate_text:
            # If no obvious text columns; try all string-typed columns
            candidate_text = [c for c in cols if isinstance(c, str)]
    if not candidate_text:
        LOG.info("No candidate text columns found in dataset %s; aborting fast prepare.", dataset_id)
        return
    id_col = '"' + filter_field.replace('"', '""') + '"'
    sel_cols = ", ".join('"' + c.replace('"', '""') + '"' for c in candidate_text)
    # Build allowed values table and join to filter
    con.execute("CREATE TEMP TABLE allowed(id VARCHAR);")
    if allowed_values:
        con.executemany("INSERT INTO allowed VALUES (?);", [(v,) for v in allowed_values])
    q = f"SELECT {id_col} AS _id, {sel_cols} FROM v JOIN allowed ON lower(CAST({id_col} AS VARCHAR)) = allowed.id"
    try:
        cur = con.execute(q)
    except Exception as exc:
        LOG.info("Fast prepare query failed, falling back: %s", exc)
        return
    # Stream in chunks
    BATCH = 2048
    while True:
        rows = cur.fetchmany(BATCH)
        if not rows:
            break
        for row in rows:
            if stats:
                stats.raw_examples += 1
            # row: (_id, col1, col2, ...)
            vals = [str(v) for v in row[1:] if isinstance(v, str) and v.strip()]
            if not vals:
                if stats:
                    stats.no_text += 1
                continue
            # emulate pick_text preference by taking the first non-empty among preferred ordering
            raw = vals[0]
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

def scan_unique_field_values(dataset_id: str, field: str, limit: int = 5000, search: Optional[str] = None,
                             fast: Optional[bool] = None, stride: int = 1) -> List[str]:
    """Return up to `limit` unique values from a metadata field in a streaming dataset.

    - Casts values to string, strips, compares in lowercase for uniqueness.
    - Intended for listing candidate books (e.g., titles or IDs) before filtering.
    """
    LOG.info("Scanning unique values for field '%s' in %s (limit=%s)", field, dataset_id, limit)
    # Prefer lightweight streaming scan by default to avoid HTTP 429.
    # Enable parquet/DuckDB acceleration only when fast=True or TE_ENABLE_FAST_SCAN is set.
    fast_env = os.getenv("TE_ENABLE_FAST_SCAN", "").strip().lower() in {"1", "true", "yes"}
    fast_enabled = bool(fast) or fast_env
    if fast_enabled:
        try:
            page_size = int(os.getenv("TE_ROWS_API_PAGE_SIZE", "200"))
        except Exception:
            page_size = 200
        try:
            max_reqs = int(os.getenv("TE_ROWS_API_MAX_REQUESTS", "20"))
        except Exception:
            max_reqs = 20
        LOG.info("Fast scan via rows API: page_size=%s max_requests=%s", page_size, max_reqs)
        res = _hf_rows_sample_distinct(dataset_id, field, limit=int(limit), page_size=page_size, max_requests=max_reqs)
        if isinstance(res, list) and res:
            return res
        # Optional duckdb fallback only if explicitly enabled
        if os.getenv("TE_ENABLE_FAST_SCAN_DUCKDB", "").strip().lower() in {"1", "true", "yes"}:
            urls = _hf_parquet_urls(dataset_id)
            if urls:
                try:
                    max_shards = int(os.getenv("TE_FAST_SCAN_MAX_SHARDS", "3"))
                except Exception:
                    max_shards = 3
                try:
                    timeout_s = float(os.getenv("TE_FAST_SCAN_TIMEOUT", "8"))
                except Exception:
                    timeout_s = 8.0
                slice_urls = urls[:max(1, max_shards)]
                LOG.info("Fast scan via duckdb: shards=%s/%s timeout=%.1fs", len(slice_urls), len(urls), timeout_s)
                def _job():
                    return _distinct_via_duckdb(slice_urls, field, limit, search)
                with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_job)
                    try:
                        duck_res = fut.result(timeout=timeout_s)
                    except Exception:
                        duck_res = None
                if isinstance(duck_res, list) and duck_res:
                    return duck_res
        LOG.info("Fast scan unavailable; falling back to streaming scan.")

    values: List[str] = []
    seen: Set[str] = set()
    try:
        ds = load_dataset(dataset_id, split="train", streaming=True)
    except Exception as exc:
        LOG.exception("Failed to open dataset %s: %s", dataset_id, exc)
        return values
    count = 0
    seen_rows = 0
    max_rows_env = os.getenv("TE_SCAN_MAX_ROWS", "").strip()
    try:
        max_rows = int(max_rows_env) if max_rows_env else 50000
    except Exception:
        max_rows = 50000
    # Precompute stride and adjust effective row budget so stride doesn't reduce variety
    try:
        s = int(stride) if stride else 1
    except Exception:
        s = 1
    s = max(1, s)
    effective_max_rows = max_rows * s if s > 1 else max_rows
    for ex in ds:
        if count >= limit:
            break
        seen_rows += 1
        # stride sampling to increase variety across grouped datasets
        if s > 1 and (seen_rows % s) != 0:
            continue
        if seen_rows > effective_max_rows:
            LOG.info("Stopping scan after %s rows (effective budget %s with stride=%s) without reaching limit %s", seen_rows, effective_max_rows, s, limit)
            break
        try:
            raw_val = ex.get(field, None)
        except Exception:
            raw_val = None
        if raw_val is None:
            continue
        if isinstance(raw_val, (list, tuple)):
            cand = [str(v).strip() for v in raw_val if v is not None]
        else:
            cand = [str(raw_val).strip()]
        for v in cand:
            if not v:
                continue
            if search:
                if search.strip().lower() not in v.lower():
                    continue
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(v)
            count += 1
            if count >= limit:
                break
    LOG.info("Scan complete: found %s unique values for %s", len(values), field)
    values.sort(key=lambda s: s.lower())
    return values

def iter_passages_streaming(dataset_id: str, split: str = "train", min_words: int = 80,
                            chunk: int = DEF_CHUNK, quote_pairs: int = 0,
                            stats: Optional[SourceStats] = None, pre_filter: bool = True,
                            filter_field: Optional[str] = None, allowed_values: Optional[Set[str]] = None,
                            text_field: Optional[str] = None):
    """Stream records without downloading full dataset; yields normalized, chunked passages."""
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for ex in ds:
        if stats:
            stats.raw_examples += 1
        # Optional per-example filter by a metadata field (e.g., title or book_id)
        if filter_field and allowed_values is not None:
            try:
                raw_val = ex.get(filter_field, None)
            except Exception:
                raw_val = None
            # normalize to lowercase string for matching
            if isinstance(raw_val, (list, tuple)):
                # include if any of the values match
                norm_vals = {str(v).strip().lower() for v in raw_val if v is not None}
                if not (norm_vals & allowed_values):
                    continue
            else:
                norm_val = str(raw_val).strip().lower() if raw_val is not None else None
                if norm_val is None or norm_val not in allowed_values:
                    continue
        # Choose text field if provided, else auto-pick
        raw = None
        if text_field:
            try:
                raw = ex.get(text_field, None)
            except Exception:
                raw = None
        if not isinstance(raw, str) or not raw.strip():
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
                    min_words: int, chunk: int, quote_pairs: int = 0, pre_filter: bool = True,
                    filter_field: Optional[str] = None, allowed_values: Optional[List[str]] = None,
                    text_field: Optional[str] = None,
                    use_fast_prepare: Optional[bool] = None) -> Tuple[Iterable[str], str, SourceStats]:
    """Return an iterator over filtered passages plus the actual dataset identifier."""
    stats = SourceStats()
    if src_mode == "HF Dataset":
        LOG.info("Streaming HF dataset '%s' (min_words=%s, chunk=%s, quote_pairs=%s, pre_filter=%s)", dataset_id, min_words, chunk, quote_pairs, pre_filter)
        allowed_set: Optional[Set[str]] = None
        if allowed_values:
            try:
                allowed_set = {str(v).strip().lower() for v in allowed_values if v is not None}
            except Exception:
                allowed_set = None

        # Attempt a fast prepare path with DuckDB over parquet when explicitly enabled
        fast_env = os.getenv("TE_ENABLE_FAST_PREP", "").strip().lower() in {"1", "true", "yes"}
        fast_enabled = bool(use_fast_prepare) or fast_env
        if fast_enabled and filter_field and allowed_set:
            urls = _hf_parquet_urls(dataset_id)
            if urls:
                LOG.info("Using fast prepare via DuckDB for %s values on field '%s'", len(allowed_set), filter_field)

                def generator_duck():
                    yielded = False
                    try:
                        for p in iter_passages_duckdb(
                            dataset_id,
                            urls,
                            filter_field,
                            allowed_set,
                            min_words,
                            chunk,
                            quote_pairs,
                            stats,
                            pre_filter,
                            text_field=text_field,
                        ):
                            yielded = True
                            yield p
                    except Exception as exc:
                        LOG.info("Fast prepare failed with error: %s; falling back to streaming.", exc)
                    if not yielded:
                        LOG.info("Fast prepare returned no rows; falling back to streaming.")
                        yield from iter_passages_streaming(
                            dataset_id,
                            split="train",
                            min_words=min_words,
                            chunk=chunk,
                            quote_pairs=quote_pairs,
                            stats=stats,
                            pre_filter=pre_filter,
                            filter_field=filter_field,
                            allowed_values=allowed_set,
                            text_field=text_field,
                        )

                return generator_duck(), dataset_id, stats
            else:
                LOG.info("No parquet URLs found; falling back to streaming filter.")

        def generator():
            yield from iter_passages_streaming(
                dataset_id,
                split="train",
                min_words=min_words,
                chunk=chunk,
                quote_pairs=quote_pairs,
                stats=stats,
                pre_filter=pre_filter,
                filter_field=filter_field,
                allowed_values=allowed_set,
                text_field=text_field,
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

def inspect_dataset_fields(dataset_id: str, split: str = "train") -> Tuple[List[str], Optional[str], Optional[str]]:
    """Return column names and guessed defaults for book/id field and text field.

    - Uses streaming to read one example for minimal overhead.
    - Defaults: prefer 'book_id', 'book', 'book_name', 'title' for id; any of CANDIDATE_TEXT_FIELDS for text.
    """
    cols: List[str] = []
    default_id: Optional[str] = None
    default_text: Optional[str] = None
    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
    except Exception as exc:
        LOG.exception("Failed to open dataset for inspection %s: %s", dataset_id, exc)
        return cols, default_id, default_text
    # attempt to get one row
    one = None
    try:
        for ex in ds:
            one = ex
            break
    except Exception as exc:
        LOG.exception("Failed to iterate dataset for inspection %s: %s", dataset_id, exc)
        return cols, default_id, default_text
    if isinstance(one, dict):
        cols = list(one.keys())
    # guess id field
    id_prefs = ["book_id", "book", "book_name", "title", "work_id", "work_title"]
    for k in id_prefs:
        if k in cols:
            default_id = k
            break
    if not default_id and cols:
        default_id = cols[0]
    # guess text field
    for k in CANDIDATE_TEXT_FIELDS:
        if k in cols:
            default_text = k
            break
    # if none match, try the longest string value in the sample
    if not default_text and isinstance(one, dict):
        longest = None
        longest_len = -1
        for k, v in one.items():
            if isinstance(v, str) and len(v) > longest_len:
                longest = k
                longest_len = len(v)
        default_text = longest
    return cols, default_id, default_text
