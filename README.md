# Medical RAG

面向 PMC OA 全文语料的医学检索增强生成项目。当前正式主线包含 153121 篇 XML、3057078 个全文 chunks、BGE-base Chroma 索引，以及 MeSH 2026 查询理解资源。

## 项目入口

- `src/medical_rag/`：可复用、可导入和可测试的核心模块。
- `scripts/`：数据分析、chunking、索引、查询理解和检索流水线的命令行入口。
- `docs/PROJECT_STRUCTURE.md`：目录职责和正式产物位置。
- `docs/NAMING_CONVENTIONS.md`：目录、代码和产物命名约定。

## 常用路径

- 原始 PMC XML：`data/raw/pmc_oa_comm/`
- MeSH 原始资源：`data/reference/mesh/2026/`
- chunk 数据集：`artifacts/datasets/chunks/`
- 正式 Chroma 索引：`artifacts/indexes/chroma/pmc_fulltext_bge_base_limit153121/`
- BM25 索引：`artifacts/indexes/bm25/`
- HuggingFace 缓存：`artifacts/models/huggingface/`
- 任务指标：`artifacts/metrics/tNNN_<task_slug>/`
- 正式报告：`reports/formal/`
- 运行日志：`logs/`
- 历史实验：`archive/experiments/`

> 原始数据、chunk 分片、Chroma 索引、模型缓存、生成后的 MeSH 词典和记录级大型统计不存储在本仓库。仓库保留源代码、结构文档、报告、图表和轻量验证产物。

## 运行方式

在项目根目录激活环境并安装本地包：

```bash
source .venv/bin/activate
python -m pip install -e . --no-deps --no-build-isolation
```

流水线入口保留在 `scripts/`。执行脚本时始终从项目根目录启动，原始数据、模型和持久化索引需在运行环境中按 `docs/PROJECT_STRUCTURE.md` 所示路径准备。

固定 `BAAI/bge-reranker-base` 比较三种融合策略：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/21_compare_fusion_with_reranker.py --output_prefix top50_q50
```

该 benchmark 对每条查询只执行一次向量检索和 BM25，再将相同候选复用于 simple、RRF 和 weighted；reranker 采用严格模式，加载或推理失败会终止实验。默认 BM25 路径是 part000 验收索引。

在已准备完整 Chroma、`BAAI/bge-reranker-base` 与全量 BM25 索引的运行环境中，使用下列参数运行正式全量对比：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/21_compare_fusion_with_reranker.py \
  --output_prefix full_bm25_top50_q50 \
  --metrics_dir artifacts/metrics/t015_full_bm25 \
  --bm25_index_dir artifacts/indexes/bm25/pmc_fulltext_bm25_limit153121 \
  --device cuda
```

全量人工评测的汇总指标位于 `artifacts/metrics/t015_full_bm25/`；原始候选文本、逐文档结果和标注预览不进入公开仓库。

`src/medical_rag/` 保存可被多个任务导入复用的核心逻辑；`scripts/` 保存命令行参数、批处理和任务编排。新增任务使用独立的 `artifacts/metrics/tNNN_<task_slug>/` 保存统计和验证结果。
