from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from medical_rag.common.pmc import ensure_output_dirs, non_empty, read_records_csv, setup_tee, write_markdown

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output_prefix", default="limit500")
    args = parser.parse_args()
    ensure_output_dirs()
    log_path = Path(f"logs/05b_build_chroma_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        records = read_records_csv(Path(args.input))
        quality = pd.read_csv(args.quality, dtype=str).fillna("")
        tokens = pd.read_csv(args.tokens, dtype=str).fillna("")
        df = records.merge(quality[["record_id", "quality_decision"]], on="record_id", how="left").merge(tokens[["record_id", "title_abstract_token_len"]], on="record_id", how="left")
        df["title_abstract_token_len"] = pd.to_numeric(df["title_abstract_token_len"], errors="coerce").fillna(0).astype(int)
        eligible = df[df["quality_decision"].isin(["keep", "keep_with_warning"]) & non_empty(df["text_title_abstract"])].copy()
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        docs = []
        manifest = []
        for _, row in eligible.iterrows():
            base_meta = {k: row.get(k, "") for k in ["record_id", "source_file", "title", "journal", "pub_year", "pmid", "pmcid", "article_type"]}
            text = row["text_title_abstract"]
            if int(row["title_abstract_token_len"]) <= 512:
                chunks = [text]
                strategy = "whole_title_abstract"
            else:
                chunks = splitter.split_text(text)
                strategy = "recursive_split_title_abstract"
            for idx, chunk in enumerate(chunks):
                meta = dict(base_meta)
                meta.update({"chunk_index": idx, "chunk_count": len(chunks), "chunk_strategy": strategy})
                docs.append(Document(page_content=chunk, metadata=meta))
                manifest.append({**meta, "chunk_chars": len(chunk)})
        persist_dir = Path("archive/experiments/indexes/chroma_limit500")
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        print(f"Eligible records: {len(eligible)}")
        print(f"Documents/chunks: {len(docs)}")
        embedding = HuggingFaceEmbeddings(model_name=MODEL, model_kwargs={'device': 'cpu'}, encode_kwargs={'batch_size': 1})
        if docs:
            Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=str(persist_dir))
        manifest_path = Path(f"artifacts/metrics/t002_corpus_analysis/chroma_{args.output_prefix}_document_manifest.csv")
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
        split_count = sum(1 for row in manifest if row["chunk_strategy"] == "recursive_split_title_abstract")
        summary = pd.DataFrame([
            {"metric": "input_records", "value": len(records)},
            {"metric": "eligible_records", "value": len(eligible)},
            {"metric": "total_chunks", "value": len(docs)},
            {"metric": "persist_directory", "value": str(persist_dir)},
            {"metric": "embedding_model", "value": MODEL},
            {"metric": "split_chunks", "value": split_count},
        ])
        summary_path = Path(f"artifacts/metrics/t002_corpus_analysis/chroma_{args.output_prefix}_build_summary.csv")
        summary.to_csv(summary_path, index=False)
        body = f"""
## 本阶段目标

在最小 embedding/Chroma 测试通过后，本阶段构建 500 篇样本范围内的 Chroma 测试库。该库用于验证批量入库流程，不等同于最终生产级 RAG 知识库。

## 入库策略

- 入库文本：`title + abstract`
- 入库记录：`quality_decision` 为 `keep` 或 `keep_with_warning`，且 `text_title_abstract` 非空
- metadata：`record_id, source_file, title, journal, pub_year, pmid, pmcid, article_type`
- 分割策略：`title+abstract <= 512 tokens` 整体入库；超过 512 tokens 的长尾样本使用 `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)`。

## 当前结果

- 输入记录数：`{len(records)}`
- 实际入库记录数：`{len(eligible)}`
- 实际写入 chunk 数：`{len(docs)}`
- 持久化目录：`{persist_dir}`

## 对后续 RAG 的影响

该库证明 500 篇摘要级数据可以进入 Chroma，并保留 metadata 供后续过滤和溯源。正式 RAG 构建时可扩展到更多文献，并根据全文 token 分析决定是否加入 body 和章节切分。
"""
        doc_path = Path(f"reports/technical/05b_chroma_{args.output_prefix}_build_notes.md")
        write_markdown(doc_path, "500 篇 Chroma 测试库构建说明", body)
        print("Wrote", summary_path)
        print("Wrote", manifest_path)
        print("Wrote", doc_path)

if __name__ == "__main__":
    main()
