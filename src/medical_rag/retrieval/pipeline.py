from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_rag.query.understanding import process_medical_query, result_to_dict
from medical_rag.retrieval.multipath import (
    DEFAULT_BM25_DIR,
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DIR,
    MultiPathRetriever,
)
from medical_rag.retrieval.reranker import BGEReranker, DEFAULT_RERANKER_MODEL
from medical_rag.retrieval.scoring import DEFAULT_CRITERIA_WEIGHTS, apply_multi_criteria_scoring
from medical_rag.retrieval.vector_store import DEFAULT_MODEL_NAME


def _normalize_fusion_relevance(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    values = [float(item.get("fusion_score") or 0.0) for item in candidates]
    low, high = min(values), max(values)
    output: list[dict[str, Any]] = []
    for item, value in zip(candidates, values):
        copied = dict(item)
        if high > low:
            copied["relevance_score"] = (value - low) / (high - low)
        else:
            copied["relevance_score"] = 1.0 if value > 0 else 0.0
        copied["reranker_raw_score"] = None
        copied["reranker_rank"] = None
        copied["reranker_model"] = None
        output.append(copied)
    return output


class RetrievalPipeline:
    def __init__(
        self,
        *,
        project_root: str | Path,
        chroma_persist_dir: str | Path = DEFAULT_CHROMA_DIR,
        chroma_collection_name: str = DEFAULT_CHROMA_COLLECTION,
        bm25_index_dir: str | Path = DEFAULT_BM25_DIR,
        embedding_model_name: str = DEFAULT_MODEL_NAME,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        terminology_path: str | Path | None = None,
        term_index_path: str | Path | None = None,
        device: str = "auto",
        reranker_batch_size: int = 8,
        reranker_max_length: int = 512,
        vector_weight: float = 0.65,
        keyword_weight: float = 0.35,
        rrf_k: int = 60,
        criteria_weights: dict[str, float] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.reranker_model_name = reranker_model_name
        self.device = device
        self.reranker_batch_size = reranker_batch_size
        self.reranker_max_length = reranker_max_length
        self.criteria_weights = dict(criteria_weights or DEFAULT_CRITERIA_WEIGHTS)
        self.terminology_path, self.term_index_path = self._terminology_paths(
            terminology_path,
            term_index_path,
        )
        self.retriever = MultiPathRetriever(
            project_root=self.project_root,
            chroma_persist_dir=chroma_persist_dir,
            chroma_collection_name=chroma_collection_name,
            bm25_index_dir=bm25_index_dir,
            embedding_model_name=embedding_model_name,
            device=device,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            rrf_k=rrf_k,
        )

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _terminology_paths(
        self,
        terminology_path: str | Path | None,
        term_index_path: str | Path | None,
    ) -> tuple[Path | None, Path | None]:
        if terminology_path or term_index_path:
            return (
                self._resolve(terminology_path) if terminology_path else None,
                self._resolve(term_index_path) if term_index_path else None,
            )
        candidates = [
            (
                self.project_root / "artifacts/terminology/mesh_2026/medical_synonyms_mesh.jsonl",
                self.project_root / "artifacts/terminology/mesh_2026/term_to_concept_mesh.json",
            ),
            (
                self.project_root / "artifacts/terminology/medical_synonyms_mesh.jsonl",
                self.project_root / "artifacts/terminology/term_to_concept_mesh.json",
            ),
        ]
        for terminology, term_index in candidates:
            if terminology.exists() and term_index.exists():
                return terminology, term_index
        return None, None

    def run(
        self,
        query: str,
        *,
        top_k_vector: int = 50,
        top_k_keyword: int = 50,
        fusion_strategy: str = "rrf",
        fusion_top_k: int = 80,
        rerank_top_k: int = 50,
        final_top_k: int = 10,
        disable_reranker: bool = False,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if final_top_k <= 0:
            raise ValueError("final_top_k must be positive")
        understood = process_medical_query(
            query,
            terminology_path=self.terminology_path,
            term_index_path=self.term_index_path,
        )
        query_info = result_to_dict(understood)
        warnings = list(query_info.get("warnings", []))
        if not understood.clean_query:
            return {
                "query": query,
                "query_info": query_info,
                "evidence": [],
                "warnings": warnings,
                "status": "invalid_query",
            }

        retrieval = self.retriever.retrieve(
            query_info,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            fusion_strategy=fusion_strategy,
            fusion_top_k=fusion_top_k,
            where=where,
        )
        warnings.extend(retrieval["warnings"])
        candidates = retrieval["results"]
        reranker_used = False
        if disable_reranker:
            ranked_by_relevance = _normalize_fusion_relevance(candidates)
            warnings.append("reranker disabled; normalized fusion_score is used as relevance_score")
        else:
            try:
                reranker = BGEReranker(
                    project_root=self.project_root,
                    model_name=self.reranker_model_name,
                    device=self.device,
                    batch_size=self.reranker_batch_size,
                    max_length=self.reranker_max_length,
                )
                ranked_by_relevance = reranker.rerank(
                    understood.clean_query,
                    candidates,
                    top_n=rerank_top_k,
                )
                reranker_used = True
            except Exception as exc:
                warnings.append(
                    f"reranker unavailable; normalized fusion_score is used instead: "
                    f"{type(exc).__name__}: {exc}"
                )
                ranked_by_relevance = _normalize_fusion_relevance(candidates)

        scored = apply_multi_criteria_scoring(
            ranked_by_relevance,
            criteria_weights=self.criteria_weights,
        )
        evidence = scored[:final_top_k]
        return {
            "query": query,
            "query_info": query_info,
            "retrieval": {key: value for key, value in retrieval.items() if key != "results"},
            "criteria_weights": self.criteria_weights,
            "reranker_requested": not disable_reranker,
            "reranker_used": reranker_used,
            "candidate_count": len(candidates),
            "scored_candidate_count": len(scored),
            "evidence": evidence,
            "warnings": warnings,
            "status": "ok" if evidence else "no_evidence",
            "scope_note": "retrieval evidence only; answer generation is outside T014",
        }
