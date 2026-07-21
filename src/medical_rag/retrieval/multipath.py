from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from medical_rag.query.understanding import BGE_QUERY_INSTRUCTION
from medical_rag.retrieval.bm25 import BM25Retriever
from medical_rag.retrieval.vector_store import (
    DEFAULT_MODEL_NAME,
    clean_metadata_value,
    encode_passages,
    load_sentence_transformer,
    resolve_device,
)


DEFAULT_CHROMA_DIR = Path("artifacts/indexes/chroma/pmc_fulltext_bge_base_limit153121")
DEFAULT_CHROMA_COLLECTION = "pmc_fulltext_bge_base_limit153121"
DEFAULT_BM25_DIR = Path("artifacts/indexes/bm25/pmc_fulltext_bm25_part000_limit153121")
FUSION_STRATEGIES = {"simple", "rrf", "weighted"}


def _as_query_dict(query_info: Any) -> dict[str, Any]:
    if is_dataclass(query_info):
        return asdict(query_info)
    if isinstance(query_info, dict):
        return query_info
    raise TypeError("query_info must be a dictionary or dataclass from the query understanding module")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _metadata_matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        clauses = where.get("$and")
        return isinstance(clauses, list) and all(_metadata_matches_where(metadata, clause) for clause in clauses)
    if "$or" in where:
        clauses = where.get("$or")
        return isinstance(clauses, list) and any(_metadata_matches_where(metadata, clause) for clause in clauses)
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            for operator, target in expected.items():
                if operator == "$eq" and actual != target:
                    return False
                if operator == "$ne" and actual == target:
                    return False
                if operator == "$in" and actual not in target:
                    return False
                if operator == "$nin" and actual in target:
                    return False
        elif str(actual) != str(expected):
            return False
    return True


def _merge_where(primary: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    if primary and override:
        return {"$and": [primary, override]}
    return override or primary


class MultiPathRetriever:
    """Retrieve from Chroma and BM25, then fuse candidates with a common schema."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        chroma_persist_dir: str | Path = DEFAULT_CHROMA_DIR,
        chroma_collection_name: str = DEFAULT_CHROMA_COLLECTION,
        bm25_index_dir: str | Path = DEFAULT_BM25_DIR,
        embedding_model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        vector_weight: float = 0.65,
        keyword_weight: float = 0.35,
        rrf_k: int = 60,
    ):
        self.project_root = Path(project_root).resolve()
        self.chroma_persist_dir = self._resolve(chroma_persist_dir)
        self.chroma_collection_name = chroma_collection_name
        self.bm25_index_dir = self._resolve(bm25_index_dir)
        self.embedding_model_name = embedding_model_name
        self.device = resolve_device(device)
        if vector_weight < 0 or keyword_weight < 0 or vector_weight + keyword_weight <= 0:
            raise ValueError("vector_weight and keyword_weight must be non-negative with a positive sum")
        weight_total = vector_weight + keyword_weight
        self.vector_weight = vector_weight / weight_total
        self.keyword_weight = keyword_weight / weight_total
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.rrf_k = rrf_k
        self._collection = None
        self._embedding_model = None
        self._bm25: BM25Retriever | None = None

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path)
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()

    def _load_vector_components(self) -> None:
        if self._collection is not None and self._embedding_model is not None:
            return
        if not self.chroma_persist_dir.exists():
            raise FileNotFoundError(f"Missing Chroma persist directory: {self.chroma_persist_dir}")
        import chromadb

        client = chromadb.PersistentClient(path=str(self.chroma_persist_dir))
        self._collection = client.get_collection(self.chroma_collection_name)
        self._embedding_model = load_sentence_transformer(
            self.embedding_model_name,
            self.device,
            self.project_root,
        )

    def _load_bm25(self) -> None:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.bm25_index_dir)

    def vector_search(
        self,
        query_info: Any,
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query = _as_query_dict(query_info)
        self._load_vector_components()
        vector_text = str(query.get("vector_query") or query.get("clean_query") or "").strip()
        instructed = str(query.get("bge_query") or "").strip()
        if not instructed:
            instructed = BGE_QUERY_INSTRUCTION + vector_text
        if not vector_text:
            raise ValueError("query_info does not contain a usable vector_query or clean_query")
        embedding = encode_passages(self._embedding_model, [instructed], batch_size=1)[0]
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        output: list[dict[str, Any]] = []
        for rank, chunk_id in enumerate(ids, start=1):
            metadata = {
                key: clean_metadata_value(value)
                for key, value in (metadatas[rank - 1] or {}).items()
            }
            distance = _safe_float(distances[rank - 1] if rank - 1 < len(distances) else None)
            output.append(
                {
                    "chunk_id": str(chunk_id),
                    "doc_id": str(metadata.get("doc_id") or ""),
                    "text": str(documents[rank - 1] or ""),
                    "metadata": metadata,
                    "retrieval_sources": ["vector"],
                    "vector_rank": rank,
                    "keyword_rank": None,
                    "vector_score": None if distance is None else 1.0 - distance,
                    "vector_distance": distance,
                    "bm25_score": None,
                    "fusion_score": None,
                    "fusion_strategy": None,
                }
            )
        return output

    def keyword_search(
        self,
        query_info: Any,
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query = _as_query_dict(query_info)
        self._load_bm25()
        fetch_k = max(top_k * 5, top_k) if where else top_k
        raw = self._bm25.search(
            query.get("keyword_query"),
            top_k=fetch_k,
            fallback_query=str(query.get("clean_query") or ""),
        )
        output: list[dict[str, Any]] = []
        for item in raw:
            if not _metadata_matches_where(item["metadata"], where):
                continue
            output.append(
                {
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "retrieval_sources": ["keyword"],
                    "vector_rank": None,
                    "keyword_rank": len(output) + 1,
                    "vector_score": None,
                    "vector_distance": None,
                    "bm25_score": item["bm25_score"],
                    "fusion_score": None,
                    "fusion_strategy": None,
                }
            )
            if len(output) >= top_k:
                break
        return output

    @staticmethod
    def _merge_candidates(
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for result in [*vector_results, *keyword_results]:
            chunk_id = result["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = dict(result)
                merged[chunk_id]["retrieval_sources"] = list(result["retrieval_sources"])
                continue
            current = merged[chunk_id]
            for field in (
                "vector_rank",
                "keyword_rank",
                "vector_score",
                "vector_distance",
                "bm25_score",
            ):
                if result.get(field) is not None:
                    current[field] = result[field]
            for source in result["retrieval_sources"]:
                if source not in current["retrieval_sources"]:
                    current["retrieval_sources"].append(source)
            if not current.get("text") and result.get("text"):
                current["text"] = result["text"]
            current["metadata"] = {**result.get("metadata", {}), **current.get("metadata", {})}
        return merged

    def fuse_results(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        *,
        fusion_strategy: str = "rrf",
        top_k: int = 80,
    ) -> list[dict[str, Any]]:
        if fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(f"fusion_strategy must be one of {sorted(FUSION_STRATEGIES)}")
        merged = self._merge_candidates(vector_results, keyword_results)
        if fusion_strategy == "simple":
            ordered_ids = list(dict.fromkeys([item["chunk_id"] for item in [*vector_results, *keyword_results]]))
            for position, chunk_id in enumerate(ordered_ids, start=1):
                merged[chunk_id]["fusion_score"] = 1.0 / position
        elif fusion_strategy == "rrf":
            for item in merged.values():
                score = 0.0
                if item.get("vector_rank") is not None:
                    score += 1.0 / (self.rrf_k + int(item["vector_rank"]))
                if item.get("keyword_rank") is not None:
                    score += 1.0 / (self.rrf_k + int(item["keyword_rank"]))
                item["fusion_score"] = score
        else:
            vector_scores = {
                item["chunk_id"]: float(item["vector_score"])
                for item in vector_results
                if item.get("vector_score") is not None
            }
            keyword_scores = {
                item["chunk_id"]: float(item["bm25_score"])
                for item in keyword_results
                if item.get("bm25_score") is not None
            }
            normalized_vector = _minmax(vector_scores)
            normalized_keyword = _minmax(keyword_scores)
            for chunk_id, item in merged.items():
                item["fusion_score"] = (
                    self.vector_weight * normalized_vector.get(chunk_id, 0.0)
                    + self.keyword_weight * normalized_keyword.get(chunk_id, 0.0)
                )
        for item in merged.values():
            item["fusion_strategy"] = fusion_strategy
        return sorted(
            merged.values(),
            key=lambda item: (
                -float(item.get("fusion_score") or 0.0),
                item.get("vector_rank") or 10**9,
                item.get("keyword_rank") or 10**9,
                item["chunk_id"],
            ),
        )[:top_k]

    def retrieve(
        self,
        query_info: Any,
        *,
        top_k_vector: int = 50,
        top_k_keyword: int = 50,
        fusion_strategy: str = "rrf",
        fusion_top_k: int = 80,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = _as_query_dict(query_info)
        effective_where = _merge_where(query.get("where_filter"), where)
        warnings: list[str] = []
        vector_results: list[dict[str, Any]] = []
        keyword_results: list[dict[str, Any]] = []
        try:
            vector_results = self.vector_search(query, top_k=top_k_vector, where=effective_where)
        except Exception as exc:
            warnings.append(f"vector retrieval failed; continuing with partial results: {type(exc).__name__}: {exc}")
        try:
            keyword_results = self.keyword_search(query, top_k=top_k_keyword, where=effective_where)
        except Exception as exc:
            warnings.append(f"keyword retrieval failed; continuing with partial results: {type(exc).__name__}: {exc}")
        if not vector_results and not keyword_results:
            warnings.append("both retrieval paths returned no candidates")
        fused = self.fuse_results(
            vector_results,
            keyword_results,
            fusion_strategy=fusion_strategy,
            top_k=fusion_top_k,
        )
        return {
            "results": fused,
            "warnings": warnings,
            "where": effective_where,
            "vector_result_count": len(vector_results),
            "keyword_result_count": len(keyword_results),
            "fused_result_count": len(fused),
            "fusion_strategy": fusion_strategy,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "rrf_k": self.rrf_k,
        }
