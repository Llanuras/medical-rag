from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from medical_rag.common.pmc import ensure_output_dirs, non_empty, read_records_csv, setup_tee, write_markdown

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERIES = ["cancer treatment clinical trial", "cardiovascular disease risk", "gene expression protein"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_prefix", default="limit500")
    args = parser.parse_args()
    ensure_output_dirs()
    log_path = Path(f"logs/05_min_embedding_chroma_test_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        df = read_records_csv(Path(args.input))
        usable = df[non_empty(df["title"]) & non_empty(df["abstract"])].head(10).copy()
        persist_dir = Path(f"archive/experiments/indexes/chroma_test_db/min_embedding_{args.output_prefix}")
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        docs = []
        for _, row in usable.iterrows():
            docs.append(Document(page_content=row["text_title_abstract"], metadata={"record_id": row["record_id"], "title": row["title"], "journal": row["journal"], "pub_year": row["pub_year"], "pmid": row["pmid"], "pmcid": row["pmcid"]}))
        print(f"Embedding model: {MODEL}")
        print(f"Sample docs: {len(docs)}")
        embedding = HuggingFaceEmbeddings(model_name=MODEL, model_kwargs={'device': 'cpu'}, encode_kwargs={'batch_size': 1})
        db = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=str(persist_dir))
        rows = []
        for query in QUERIES:
            results = db.similarity_search(query, k=min(3, len(docs)))
            for rank, doc in enumerate(results, 1):
                rows.append({"query": query, "rank": rank, "record_id": doc.metadata.get("record_id", ""), "title": doc.metadata.get("title", ""), "journal": doc.metadata.get("journal", ""), "pub_year": doc.metadata.get("pub_year", ""), "preview": doc.page_content[:300]})
                print(query, rank, doc.metadata.get("record_id", ""), doc.metadata.get("title", "")[:80])
        result_path = Path(f"artifacts/metrics/t002_corpus_analysis/min_embedding_chroma_test_results_{args.output_prefix}.csv")
        pd.DataFrame(rows).to_csv(result_path, index=False)
        body = f"""
## 1. 本阶段目标

本阶段只验证 embedding + Chroma pipeline 是否可行，不做正式大规模建库。

## 2. 为什么需要 embedding 测试

后续 RAG 需要将文本转为向量，并写入向量数据库，才能执行 similarity_search。因此需要先用少量样本确认链路可用。

## 3. 为什么只用 5-10 条样本

本次任务重点是数据分析和策略设计，不是大规模向量化。选择 10 条样本可以验证模型下载、向量生成、Chroma 写入和检索，同时控制耗时和复杂度。

## 4. 可选策略

- 不做 embedding 测试：速度最快，但无法确认向量库链路可用。
- 最小 embedding 测试：能验证关键链路，成本低，是当前选择。
- 对 500 篇全部 embedding：可生成测试库，但应在最小测试通过后执行。
- 对全文全部 embedding：成本更高，且需要先明确全文切分策略。

## 5. 测试结果

使用 `{MODEL}` 对 `{len(docs)}` 条样本文本写入 Chroma，并执行 `{len(QUERIES)}` 个查询。真实检索结果保存在 `artifacts/metrics/t002_corpus_analysis/min_embedding_chroma_test_results_{args.output_prefix}.csv`。

## 6. 后续扩展建议

正式 RAG 构建时应根据 token 长度分析决定是否切分，再批量 embedding 并写入 Chroma。
"""
        doc_path = Path(f"reports/technical/05_embedding_chroma_test_notes_{args.output_prefix}.md")
        write_markdown(doc_path, "最小 embedding/Chroma 可用性测试说明", body)
        print("Wrote", result_path)
        print("Wrote", doc_path)

if __name__ == "__main__":
    main()
