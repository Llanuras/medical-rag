from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from medical_rag.retrieval.vector_store import (
    ensure_project_hf_cache,
    find_local_model_snapshot,
    resolve_device,
)


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class RerankerUnavailableError(RuntimeError):
    pass


class BGEReranker:
    """Cross-encoder reranker for a bounded candidate list, never for the full corpus."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.project_root = Path(project_root).resolve()
        self.model_name = model_name
        self.device = resolve_device(device)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.model_source: str | None = None

    def _resolve_model_source(self) -> Path:
        candidate = Path(self.model_name)
        if candidate.exists():
            return candidate.resolve()
        snapshot = find_local_model_snapshot(self.project_root, self.model_name)
        if snapshot is not None:
            return snapshot
        raise RerankerUnavailableError(
            f"Reranker model {self.model_name!r} was not found in "
            f"{self.project_root / 'artifacts/models/huggingface'}. "
            "Run the approved manual hf download command or use --disable_reranker."
        )

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        ensure_project_hf_cache(self.project_root)
        source = self._resolve_model_source()
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(source),
                local_files_only=True,
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_source = str(source)
        except Exception as exc:
            self.tokenizer = None
            self.model = None
            raise RerankerUnavailableError(
                f"Failed to load reranker from local source {source}: {type(exc).__name__}: {exc}"
            ) from exc

    def _score_batch(self, query_text: str, passages: list[str]) -> tuple[list[float], list[float]]:
        import torch

        encoded = self.tokenizer(
            [query_text] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
        if logits.ndim == 1:
            raw = logits
        elif logits.shape[-1] == 1:
            raw = logits[:, 0]
        else:
            raw = logits[:, -1]
        raw_scores = [float(value) for value in raw.detach().cpu().tolist()]
        relevance_scores = [1.0 / (1.0 + math.exp(-max(min(value, 60.0), -60.0))) for value in raw_scores]
        return raw_scores, relevance_scores

    def rerank(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        *,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("query_text must be non-empty")
        self.load()
        limit = len(candidates) if top_n is None else max(0, min(top_n, len(candidates)))
        selected = [dict(item) for item in candidates[:limit]]
        for start in range(0, len(selected), self.batch_size):
            batch = selected[start : start + self.batch_size]
            passages = [str(item.get("text") or "") for item in batch]
            raw_scores, relevance_scores = self._score_batch(query_text, passages)
            for item, raw_score, relevance_score in zip(batch, raw_scores, relevance_scores):
                item["reranker_raw_score"] = raw_score
                item["relevance_score"] = relevance_score
                item["reranker_model"] = self.model_name
        ranked = sorted(
            selected,
            key=lambda item: (-float(item.get("relevance_score") or 0.0), item.get("chunk_id", "")),
        )
        for rank, item in enumerate(ranked, start=1):
            item["reranker_rank"] = rank
        return ranked
