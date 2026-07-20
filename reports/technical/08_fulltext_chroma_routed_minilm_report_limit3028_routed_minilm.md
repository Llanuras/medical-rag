# 3028 篇全文 routed chunk + all-MiniLM Chroma 规模验证

## 实验口径

本次为正式 3028 篇全文 chunk + Chroma 规模验证，严格按照既有全文结构分析结果做三路由：

- `whole_document_under_512`：`28` 篇，全文不分割。
- `semantic_section`：`2695` 篇，按 XML 顶层章节语义切分，并保留 `section_title` metadata；超长章节只在章节内部递归切分。
- `recursive_fallback_no_section`：`305` 篇，无章节且全文超过 512 tokens，使用 `RecursiveCharacterTextSplitter` 兜底。

Embedding 使用项目缓存中的 `sentence-transformers/all-MiniLM-L6-v2`，`HF_HOME=/Users/plain/Documents/实习/百度大模型/medical_rag/artifacts/models/huggingface`。

## 实测结果

- 实际写入 chunks：`74276`
- Chroma collection count：`74276`
- chunks/篇 mean：`24.53`
- chunks/篇 median：`24.00`
- chunks/篇 p95：`47.00`
- chunk token p95：`401.00`
- 切分耗时：`241.88` 秒
- 模型加载耗时：`0.16` 秒
- Chroma 写入耗时：`905.92` 秒
- 查询耗时：`0.02` 秒
- 总耗时：`1149.14` 秒
- 向量库大小：`985.91` MB
- Chroma 目录：`archive/experiments/indexes/chroma_fulltext_limit3028_routed_minilm`

## 检索 sanity check

- `Plasmodium falciparum intraerythrocytic developmental cycle transcriptome` -> `pmc_000001` / Introduction: . 2002 ). Although ascribing putative roles for these ORFs in the absence of sequence similarity remains challenging, their unique nature may be key to identifying Plasmodium -specific pathways as candidates for antimalarial strategies. The complete P. falciparum lifecycle encompasses three major developmental stages: 
- `pRb inactivation mammary cells tumor initiation progression` -> `pmc_000089` / Full text fallback: . Building on this work, Terry Van Dyke and colleagues report that loss of the pRb tumor suppressor in mammary tissue has the same effect—predisposition to tumor formation—seen in these other cell types. Despite the different environment inherent in each cell type, the initial events following loss of the pRb pathway w
- `type 2 diabetes high protein diet insulin concentration` -> `pmc_001051` / Results: . However, in the subjects with type 2 diabetes, the glucose concentration actually decreased over the 5 hours of that study (Figure 2 ). Figure 2 Glucose (left panel) and insulin (right panel) response to ingestion of 50 g of protein in the form of lean beef. Data from 8 non-diabetic subjects (white lines, bottom) and
- `immune response B cells master regulator` -> `pmc_000257` / Full text fallback: Training the Immune Response: B-cells' Master Regulator
- `SARS coronavirus spike protein trafficking` -> `pmc_002582` / Title and abstract: The Severe Acute Respiratory Syndrome (SARS)-coronavirus 3a protein may function as a modulator of the trafficking properties of the spike protein Background A recent publication reported that a tyrosine-dependent sorting signal, present in cytoplasmic tail of the spike protein of most coronaviruses, mediates the intra

完整 top-5 结果见 `artifacts/metrics/t005_routed_minilm/fulltext_chroma_routed_minilm_query_results_limit3028_routed_minilm.csv`。

## 关于旧 07 报告

`reports/technical` 中多个 `07_fulltext_chroma_scale_test_*` 文件是此前错误口径和冒烟实验生成的中间文件，包含 hash/TF-IDF embedding 和非严格三路由切分，不应作为本阶段正式结论。正式结论以本报告和 `artifacts/metrics/fulltext_chroma_routed_minilm_*_limit3028_routed_minilm.csv` 为准。
