# 项目结构

```text
medical_rag/
├── README.md
├── pyproject.toml
├── src/medical_rag/                 # 可复用 Python 包
│   ├── common/                      # PMC 解析和通用工具
│   ├── embeddings/                  # embedding 辅助实现
│   ├── query/                       # 查询理解与术语匹配
│   └── retrieval/                   # 向量模型和 Chroma 公共逻辑
├── scripts/                         # 可直接执行的流水线入口
├── data/
│   ├── raw/pmc_oa_comm/             # 原始 PMC OA XML
│   └── reference/mesh/2026/         # 官方 MeSH 2026 XML.GZ
├── artifacts/
│   ├── datasets/records/            # 解析后的文档记录
│   ├── datasets/chunks/             # 正式 chunk 分片与 manifest
│   ├── indexes/chroma/              # 正式 Chroma 持久化索引
│   ├── metrics/                     # 按任务编号归档的 CSV/JSON 统计和验证结果
│   │   ├── t001_environment/        # 环境准备与验证指标
│   │   ├── t002_corpus_analysis/    # limit500/3028 语料分析与历史检索指标
│   │   ├── t005_routed_minilm/      # routed MiniLM 验证指标
│   │   ├── t006_fullscale_analysis/ # 153121 全量语料分析指标
│   │   ├── t007_chunking/           # chunk 处理与质量指标
│   │   ├── t008_vector_index/       # BGE 索引构建与验证指标
│   │   ├── t009_query_understanding/ # 静态查询理解测试指标
│   │   └── t010_mesh_query_understanding/ # MeSH 查询理解指标
│   ├── models/                      # 项目级模型缓存
│   └── terminology/                 # 生成后的术语资源
├── reports/
│   ├── formal/                      # 正式阶段报告和 mentor 交付
│   ├── technical/                   # 技术说明与分析笔记
│   ├── figures/                     # 报告图表
│   ├── samples/                     # 人工审阅样本
│   └── assets/                      # 截图等静态资源
├── logs/                            # 运行日志
├── releases/                        # 可分发的轻量发布包
├── archive/experiments/             # 历史基线和探索性实验
└── runtime/cache/                   # 可再生的运行时缓存
```

## 正式主线

正式数据、索引和验证结果使用 `limit153121` 范围标识。`limit500`、`limit3028`、`smoke20` 和 `part000` 仅代表基线、验收或局部验证，不作为全量结论。

大型原始数据、Parquet 分片、模型缓存和 Chroma 索引保存在独立运行环境，不进入 Git 仓库。

## `src` 与 `scripts` 的边界

- `src/medical_rag/`：可复用、可导入、可单元测试的业务逻辑，例如 PMC 解析工具、查询理解和向量库公共函数。
- `scripts/`：面向具体任务的命令行入口，负责参数解析、批处理、进度控制和调用 `src/medical_rag/`。
- 如果一段逻辑会被两个以上脚本复用，应进入 `src/medical_rag/`，脚本中只保留任务编排。

## 任务指标目录

`artifacts/metrics/` 根目录不得直接存放文件。每个新增任务建立独立 `TNNN` 编号和 `artifacts/metrics/tNNN_<task_slug>/` 目录；代码、报告和验证产物必须引用同一个正式路径。
