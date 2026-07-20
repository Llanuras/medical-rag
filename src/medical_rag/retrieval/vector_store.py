from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_METADATA_FIELDS = [
    "doc_id",
    "chunk_index",
    "total_chunks",
    "source_title",
    "journal",
    "pub_date",
    "pub_year",
    "pmid",
    "pmcid",
    "article_type",
    "section_title",
    "section_title_norm",
    "split_strategy",
    "quality_decision",
    "source_file",
    "token_count",
]


class Tee:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.log = log_path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.flush()
        self.log.close()


def project_root_from_script(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[1]


def ensure_output_dirs(project_root: Path) -> None:
    for rel in ["artifacts/metrics/t008_vector_index", "logs", "reports/formal", "artifacts/indexes/chroma"]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)


def ensure_project_hf_cache(project_root: Path) -> Path:
    hf_home = project_root / "artifacts/models/huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return hf_home


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_local_model_snapshot(project_root: Path, model_name: str) -> Path | None:
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots_dirs = [
        project_root / "artifacts/models/huggingface" / "hub" / cache_name / "snapshots",
        project_root / "artifacts/models/huggingface" / cache_name / "snapshots",
    ]
    candidates = []
    for snapshots_dir in snapshots_dirs:
        if snapshots_dir.exists():
            candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())
    for snapshot in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True):
        if (snapshot / "modules.json").exists() or (snapshot / "config.json").exists():
            return snapshot
    return None


def load_sentence_transformer(model_name: str, device: str, project_root: Path):
    ensure_project_hf_cache(project_root)
    from sentence_transformers import SentenceTransformer

    snapshot = find_local_model_snapshot(project_root, model_name)
    if snapshot is not None:
        print(f"[MODEL_LOAD] local snapshot: {snapshot}")
        return SentenceTransformer(
            str(snapshot),
            device=device,
            cache_folder=str(project_root / "artifacts/models/huggingface"),
            local_files_only=True,
        )

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[MODEL_LOAD] model name download: {model_name}")
    print(f"[MODEL_LOAD] HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    return SentenceTransformer(
        model_name,
        device=device,
        cache_folder=str(project_root / "artifacts/models/huggingface"),
    )


def encode_passages(model, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32").tolist()


def encode_queries(model, queries: list[str], batch_size: int) -> list[list[float]]:
    instructed = [BGE_QUERY_INSTRUCTION + q for q in queries]
    return encode_passages(model, instructed, batch_size=batch_size)


def clean_metadata_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    try:
        import numpy as np
        import pandas as pd

        if pd.isna(value):
            return ""
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            if math.isnan(float(value)):
                return ""
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def clean_metadata(row: dict[str, Any], fields: Iterable[str] = DEFAULT_METADATA_FIELDS) -> dict[str, Any]:
    return {field: clean_metadata_value(row.get(field, "")) for field in fields}


def parse_part_filter(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    selected: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(item))
    return selected


def parse_where_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--where_json must decode to a JSON object")
    return data


def truncate_query(query: str, max_chars: int = 4000) -> str:
    query = " ".join(query.split())
    if len(query) > max_chars:
        return query[:max_chars]
    return query


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def elapsed_rate(done: int, total: int, start_time: float) -> tuple[float, str]:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = done / elapsed
    if rate <= 0 or done <= 0 or total <= 0:
        return rate, "unknown"
    remaining = max(total - done, 0) / rate
    return rate, human_seconds(remaining)
