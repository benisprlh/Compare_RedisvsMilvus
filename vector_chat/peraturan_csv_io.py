"""Shared CSV parsing for peraturan loaders (Redis, Milvus, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

# Project root: vector_chat/ -> parent parent
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "list.csv" / "list.csv"

CONTENT_COLUMNS = [
    "Title",
    "wish_bt",
    "wrapper_URL",
    "wrapper",
    "loc_open",
    "Singkatan Jenis/Bentuk Peraturan",
    "Nomor",
    "Tahun",
    "Tentang",
    "Pemrakarsa",
    "Tempat Penetapan",
    "Ditetapkan Tanggal",
    "Pejabat Penetapan",
    "Diundangkan Tanggal",
    "Pejabat Pengundangan",
    "Tanggal Berlaku",
    "Status",
]

MAX_EMBED_CHARS = 12000


def _clean_cell(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return " ".join(s.split())


def row_to_content(row: Dict[str, Any]) -> str:
    lines: List[str] = []
    for col in CONTENT_COLUMNS:
        if col not in row:
            continue
        v = _clean_cell(row[col])
        if v:
            lines.append(f"{col}: {v}")
    text = "\n".join(lines)
    if len(text) > MAX_EMBED_CHARS:
        text = text[:MAX_EMBED_CHARS]
    return text


def iter_csv_chunks(
    path: Path, chunksize: int, limit: Optional[int]
) -> Iterator[pd.DataFrame]:
    total = 0
    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    for chunk in reader:
        if limit is not None:
            remaining = limit - total
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        total += len(chunk)
        yield chunk
        if limit is not None and total >= limit:
            break
