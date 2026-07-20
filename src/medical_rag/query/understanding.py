from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_QUERY_CHARS = 1000
BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "
MEDICAL_SYNONYMS = {
    "mi": ["myocardial infarction", "heart attack"], "myocardial infarction": ["mi", "heart attack"],
    "heart attack": ["myocardial infarction", "mi"], "metformin": ["dimethylbiguanide", "glucophage"],
    "二甲双胍": ["metformin"], "cardiovascular disease": ["CVD", "cardiovascular disorder", "cardiovascular outcome"],
    "心血管疾病": ["cardiovascular disease", "CVD", "cardiovascular outcome"], "diabetes": ["diabetes mellitus"],
    "type 2 diabetes": ["type 2 diabetes mellitus", "T2DM"], "egfr": ["epidermal growth factor receptor"],
    "pcr": ["polymerase chain reaction"], "hiv": ["human immunodeficiency virus"],
    "sars": ["severe acute respiratory syndrome"], "aspirin": ["acetylsalicylic acid"],
    "warfarin": ["coumadin"], "insulin": ["insulin therapy"],
}


@dataclass(frozen=True)
class EntityDefinition:
    phrase: str
    normalized: str
    entity_type: str


@dataclass
class MedicalEntity:
    text: str
    normalized: str
    entity_type: str
    start: int
    end: int
    synonyms: list[str]
    concept_id: str | None = None
    source: str = "fallback_seed"
    mesh_record_type: str | None = None
    tree_numbers: list[str] | None = None


@dataclass
class QueryUnderstandingResult:
    raw_query: str
    clean_query: str
    entities: list[MedicalEntity]
    expanded_terms: list[str]
    vector_query: str
    bge_query: str
    keyword_query: dict[str, Any]
    where_filter: dict[str, Any] | None
    filter_plan: dict[str, Any]
    warnings: list[str]


@dataclass
class MeshResources:
    concepts: dict[str, dict[str, Any]]
    term_index: dict[str, list[dict[str, Any]]]
    terminology_path: str
    term_index_path: str


_RAW_DEFS = [
    ("metformin", "metformin", "drug"), ("二甲双胍", "metformin", "drug"), ("aspirin", "aspirin", "drug"), ("阿司匹林", "aspirin", "drug"), ("atorvastatin", "atorvastatin", "drug"), ("他汀", "statin", "drug"), ("warfarin", "warfarin", "drug"), ("华法林", "warfarin", "drug"), ("insulin", "insulin", "drug"), ("胰岛素", "insulin", "drug"),
    ("myocardial infarction", "myocardial infarction", "disease"), ("heart attack", "myocardial infarction", "disease"), ("心肌梗死", "myocardial infarction", "disease"), ("心脏病发作", "myocardial infarction", "disease"), ("MI", "myocardial infarction", "disease"), ("cardiovascular disease", "cardiovascular disease", "disease"), ("心血管疾病", "cardiovascular disease", "disease"), ("type 2 diabetes", "type 2 diabetes", "disease"), ("2型糖尿病", "type 2 diabetes", "disease"), ("diabetes", "diabetes", "disease"), ("糖尿病", "diabetes", "disease"), ("lung cancer", "lung cancer", "disease"), ("肺癌", "lung cancer", "disease"), ("breast cancer", "breast cancer", "disease"), ("乳腺癌", "breast cancer", "disease"), ("HIV", "HIV", "disease"), ("SARS", "SARS", "disease"),
    ("EGFR", "EGFR", "gene/protein"), ("BRCA1", "BRCA1", "gene/protein"), ("ACE2", "ACE2", "gene/protein"),
    ("polymerase chain reaction", "polymerase chain reaction", "method"), ("PCR", "PCR", "method"), ("DNA amplification", "DNA amplification", "method"),
    ("insulin sensitivity", "insulin sensitivity", "outcome"), ("cardiovascular outcome", "cardiovascular outcome", "outcome"), ("adverse event", "adverse event", "outcome"), ("mortality", "mortality", "outcome"), ("survival", "survival", "outcome"), ("resistance", "resistance", "outcome"), ("bleeding risk", "bleeding risk", "outcome"), ("gene expression", "gene expression", "outcome"), ("reverse transcriptase inhibitor", "reverse transcriptase inhibitor", "outcome"), ("mutation", "mutation", "outcome"), ("treatment", "treatment", "outcome"), ("coronavirus spike protein", "coronavirus spike protein", "outcome"),
]
ENTITY_DEFINITIONS = [EntityDefinition(*item) for item in _RAW_DEFS]
PUNCT = str.maketrans({"，": ",", "。": ".", "？": "?", "！": "!", "；": ";", "：": ":", "（": "(", "）": ")", "、": ",", "　": " "})
_CACHE: dict[tuple[str, str], MeshResources] = {}


def _dedupe(values: list[str]) -> list[str]:
    result, seen = [], set()
    for value in values:
        value = value.strip()
        if value and value.casefold() not in seen:
            result.append(value); seen.add(value.casefold())
    return result


def clean_query(query: str | None) -> str:
    if query is None:
        return ""
    value = re.sub(r"\s+", " ", str(query).translate(PUNCT).strip())
    return re.sub(r"[?.!,;:]+$", "", re.sub(r"([?!,.;:])\1+", r"\1", value)).strip()[:MAX_QUERY_CHARS].strip()


def _pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.I) if re.search(r"[A-Za-z0-9]", phrase) else re.compile(escaped)


def _fallback_synonyms(item: EntityDefinition) -> list[str]:
    values = MEDICAL_SYNONYMS.get(item.phrase.casefold(), []) + MEDICAL_SYNONYMS.get(item.normalized.casefold(), [])
    return [value for value in _dedupe(values) if value.casefold() != item.normalized.casefold()][:5]


def find_static_entities(query: str) -> list[MedicalEntity]:
    candidates = []
    for item in sorted(ENTITY_DEFINITIONS, key=lambda x: len(x.phrase), reverse=True):
        candidates.extend((match.start(), match.end(), item, match.group(0)) for match in _pattern(item.phrase).finditer(query))
    return _select_static(candidates)


def _select_static(candidates: list[tuple[int, int, EntityDefinition, str]]) -> list[MedicalEntity]:
    output, occupied, seen = [], [], set()
    for start, end, item, text in sorted(candidates, key=lambda x: (x[0], -(x[1] - x[0]))):
        key = (item.entity_type, item.normalized.casefold())
        if key in seen or any(start < right and end > left for left, right in occupied):
            continue
        output.append(MedicalEntity(text, item.normalized, item.entity_type, start, end, _fallback_synonyms(item)))
        occupied.append((start, end)); seen.add(key)
    return output


def _read_concepts(path: Path) -> dict[str, dict[str, Any]]:
    concepts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            concepts[item["concept_id"]] = item
    return concepts


def load_mesh_resources(terminology_path: str | Path | None, term_index_path: str | Path | None) -> MeshResources | None:
    if not terminology_path or not term_index_path:
        return None
    terms, index = Path(terminology_path), Path(term_index_path)
    if not terms.exists() or not index.exists():
        return None
    key = (str(terms.resolve()), str(index.resolve()))
    if key not in _CACHE:
        with index.open(encoding="utf-8") as handle:
            term_index = json.load(handle)
        _CACHE[key] = MeshResources(_read_concepts(terms), term_index, key[0], key[1])
    return _CACHE[key]


def _query_ngrams(query: str, max_words: int = 8) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", query.casefold())
    output = {query.casefold()}
    for size in range(1, min(max_words, len(tokens)) + 1):
        for start in range(len(tokens) - size + 1):
            output.add(" ".join(tokens[start:start + size]))
    return output


def find_mesh_entities(query: str, resources: MeshResources) -> list[MedicalEntity]:
    candidates = []
    for key in _query_ngrams(query):
        for hit in resources.term_index.get(key, []):
            for match in _pattern(key).finditer(query):
                concept = resources.concepts.get(hit["concept_id"], {})
                synonyms = [term for term in concept.get("terms", []) if term.casefold() != match.group(0).casefold() and 2 <= len(term) <= 100][:5]
                candidates.append((match.start(), match.end(), hit, match.group(0), concept, synonyms))
    output, occupied, seen = [], [], set()
    for start, end, hit, text, concept, synonyms in sorted(candidates, key=lambda x: (x[0], -(x[1] - x[0]))):
        concept_id = hit["concept_id"]
        if concept_id in seen or any(start < right and end > left for left, right in occupied):
            continue
        record_type = hit.get("mesh_record_type", "descriptor")
        output.append(MedicalEntity(text, hit.get("preferred_term", text), f"mesh_{record_type}", start, end, _dedupe(synonyms)[:5], concept_id, "MeSH", record_type, hit.get("tree_numbers", [])))
        occupied.append((start, end)); seen.add(concept_id)
    return output


def merge_entities(mesh: list[MedicalEntity], fallback: list[MedicalEntity]) -> list[MedicalEntity]:
    output, occupied, seen = [], [], set()
    for entity in mesh + fallback:
        key = entity.concept_id or (entity.entity_type, entity.normalized.casefold())
        if key in seen or any(entity.start < right and entity.end > left for left, right in occupied):
            continue
        output.append(entity); occupied.append((entity.start, entity.end)); seen.add(key)
    return sorted(output, key=lambda item: item.start)


def build_keyword_query(entities: list[MedicalEntity], expanded: list[str]) -> dict[str, Any]:
    buckets = {"drug": [], "disease": [], "gene_protein": [], "method": [], "outcome": []}
    names = {"drug": "drug", "disease": "disease", "gene/protein": "gene_protein", "method": "method", "outcome": "outcome"}
    for entity in entities:
        bucket = names.get(entity.entity_type, "disease")
        buckets[bucket] = _dedupe(buckets[bucket] + [entity.normalized])
    required = _dedupe(buckets["drug"] + buckets["disease"] + buckets["gene_protein"])
    required_keys = {item.casefold() for item in required}
    return {"required_terms": required, "optional_terms": [item for item in expanded if item.casefold() not in required_keys], "entity_terms": buckets}


def extract_metadata_filters(query: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    lower, clauses, plan = query.casefold(), [], {}
    ranged = re.search(r"\b(?:from\s+)?(19\d{2}|20\d{2})\s+(?:to|-)\s+(19\d{2}|20\d{2})\b", lower)
    after, before = re.search(r"\b(?:after|since)\s+(19\d{2}|20\d{2})\b", lower), re.search(r"\bbefore\s+(19\d{2}|20\d{2})\b", lower)
    if ranged: plan.update(pub_year_gte=ranged.group(1), pub_year_lte=ranged.group(2))
    elif after: plan["pub_year_gte"] = after.group(1)
    elif before: plan["pub_year_lte"] = before.group(1)
    else:
        year = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?:年)?(?!\d)", query)
        if year: clauses.append({"pub_year": year.group(1)}); plan["pub_year_eq"] = year.group(1)
    if "pub_year_gte" in plan or "pub_year_lte" in plan:
        plan["where_compatibility_note"] = "pub_year is a Chroma string; apply range after retrieval or verify string-range support."
    for journal, pattern in (("PLoS ONE", r"\bplos\s+one\b"), ("Nature", r"\bnature\b"), ("BMC", r"\bbmc\b")):
        if re.search(pattern, lower): clauses.append({"journal": journal}); plan["journal"] = journal; break
    if re.search(r"\bresearch\s+article\b", lower): clauses.append({"article_type": "research-article"}); plan["article_type"] = "research-article"
    elif re.search(r"\breview\b", lower): plan["article_type"] = "review-article"; plan["article_type_note"] = "Confirm review labels before Chroma filtering."
    sections = [name for name in ("methods", "results", "discussion", "conclusion") if re.search(rf"\b{name}\b", lower)]
    if sections: plan["section_title_norm"] = sections; plan["section_note"] = "Confirm normalized labels before Chroma filtering."
    return (None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}), plan


def process_medical_query(query: str | None, terminology_path: str | Path | None = None, term_index_path: str | Path | None = None, mesh_resources: MeshResources | None = None) -> QueryUnderstandingResult:
    raw = "" if query is None else str(query)
    warnings = [f"query exceeded {MAX_QUERY_CHARS} characters and was truncated"] if len(raw) > MAX_QUERY_CHARS else []
    cleaned = clean_query(raw)
    if not cleaned:
        return QueryUnderstandingResult(raw, "", [], [], "", "", build_keyword_query([], []), None, {}, ["empty query: provide a non-empty medical question"])
    resources = mesh_resources or load_mesh_resources(terminology_path, term_index_path)
    requested = bool(terminology_path or term_index_path)
    if requested and resources is None:
        warnings.append("MeSH terminology/index unavailable; static fallback was used")
    mesh_entities = find_mesh_entities(cleaned, resources) if resources else []
    entities = merge_entities(mesh_entities, find_static_entities(cleaned))
    expanded = _dedupe([term for entity in entities for term in [entity.normalized, *entity.synonyms]])
    vector = " ".join(_dedupe([term for entity in entities for term in (entity.text, entity.normalized)] + expanded))
    if not vector: vector = cleaned; warnings.append("no medical entity matched; using cleaned query unchanged")
    where_filter, filter_plan = extract_metadata_filters(cleaned)
    return QueryUnderstandingResult(raw, cleaned, entities, expanded, vector, BGE_QUERY_INSTRUCTION + vector, build_keyword_query(entities, expanded), where_filter, filter_plan, warnings)


def result_to_dict(result: QueryUnderstandingResult) -> dict[str, Any]:
    return asdict(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="MeSH-prioritized medical query understanding with static fallback")
    parser.add_argument("--query", required=True); parser.add_argument("--terminology"); parser.add_argument("--term_index"); parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = process_medical_query(args.query, args.terminology, args.term_index)
    payload = result_to_dict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}" for key, value in payload.items()))
    return 2 if not result.clean_query else 0


if __name__ == "__main__":
    raise SystemExit(main())
