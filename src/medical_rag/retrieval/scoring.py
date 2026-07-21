from __future__ import annotations

import re
from datetime import datetime
from typing import Any


DEFAULT_CRITERIA_WEIGHTS = {
    "relevance": 0.60,
    "recency": 0.25,
    "authority": 0.15,
}

AUTHORITY_PRIORS = {
    "plos biology": 0.90,
    "plos medicine": 0.90,
    "british journal of cancer": 0.85,
    "emerging infectious diseases": 0.85,
    "environmental health perspectives": 0.85,
    "bmc medicine": 0.80,
    "bmc biology": 0.80,
    "bmc bioinformatics": 0.75,
    "bmc genomics": 0.75,
    "bmc cancer": 0.75,
    "plos one": 0.75,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_year(value: Any) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def recency_score(
    pub_year: Any,
    *,
    current_year: int | None = None,
    decay_years: int = 20,
    missing_score: float = 0.50,
) -> float:
    """Linear soft prior; it never filters a document out."""
    if decay_years <= 0:
        raise ValueError("decay_years must be positive")
    year = _extract_year(pub_year)
    if year is None:
        return _clamp(missing_score)
    now = current_year or datetime.now().year
    age = max(now - year, 0)
    return _clamp(1.0 - age / decay_years)


def authority_score(journal: Any, *, default_score: float = 0.50) -> float:
    """Return an auditable journal prior, not an impact factor."""
    normalized = " ".join(str(journal or "").casefold().split())
    return _clamp(AUTHORITY_PRIORS.get(normalized, default_score))


def final_score(
    relevance: float,
    recency: float,
    authority: float,
    *,
    criteria_weights: dict[str, float] | None = None,
) -> float:
    weights = dict(DEFAULT_CRITERIA_WEIGHTS if criteria_weights is None else criteria_weights)
    required = {"relevance", "recency", "authority"}
    missing = required - set(weights)
    if missing:
        raise ValueError(f"criteria_weights missing keys: {sorted(missing)}")
    if any(float(weights[key]) < 0 for key in required):
        raise ValueError("criteria weights must be non-negative")
    total = sum(float(weights[key]) for key in required)
    if total <= 0:
        raise ValueError("criteria weights must sum to a positive value")
    return _clamp(
        (
            float(weights["relevance"]) * _clamp(relevance)
            + float(weights["recency"]) * _clamp(recency)
            + float(weights["authority"]) * _clamp(authority)
        )
        / total
    )


def apply_multi_criteria_scoring(
    candidates: list[dict[str, Any]],
    *,
    criteria_weights: dict[str, float] | None = None,
    current_year: int | None = None,
    decay_years: int = 20,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        metadata = item.get("metadata") or {}
        relevance = _clamp(float(item.get("relevance_score") or 0.0))
        recency = recency_score(
            metadata.get("pub_year") or metadata.get("pub_date"),
            current_year=current_year,
            decay_years=decay_years,
        )
        authority = authority_score(metadata.get("journal"))
        item["relevance_score"] = relevance
        item["recency_score"] = recency
        item["authority_score"] = authority
        item["final_score"] = final_score(
            relevance,
            recency,
            authority,
            criteria_weights=criteria_weights,
        )
        output.append(item)
    ranked = sorted(
        output,
        key=lambda item: (
            -float(item["final_score"]),
            -float(item["relevance_score"]),
            item.get("chunk_id", ""),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["final_rank"] = rank
    return ranked
