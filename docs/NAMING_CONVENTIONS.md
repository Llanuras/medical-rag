# 命名约定

## 代码

- Python 文件、模块、变量统一使用 `lower_snake_case`。
- `scripts/` 中数字前缀表示不可变的历史流水线阶段；新增阶段使用下一个整数，不复用旧编号。
- 可复用实现放入 `src/medical_rag/`，流水线入口只保留参数解析和编排逻辑。

## 数据与实验范围

- `limit153121`：正式全量语料范围。
- `limit3028`、`limit500`：历史基线或阶段性验证。
- `smoke20`：冒烟测试。
- `test3028_pmc000`：固定验收集。
- `partNNN`：分片或局部索引，必须使用三位补零。

## 产物

每个任务的机器可读统计和验证产物必须放入：

```text
artifacts/metrics/tNNN_<task_slug>/
```

例如：

```text
artifacts/metrics/t007_chunking/chunk_summary_limit153121.csv
artifacts/metrics/t008_vector_index/vector_index_stats_limit153121_bge_base.json
artifacts/metrics/t010_mesh_query_understanding/query_understanding_examples_mesh.csv
```

禁止直接向 `artifacts/metrics/` 根目录写文件。一个任务只能写入自己的 metrics 目录；跨任务读取必须显式使用来源任务路径。

机器可读产物使用：

```text
<subject>_<artifact>_<scope>_<model_or_method>.<ext>
```

例如：

```text
pmc_chunks_limit153121_part000.parquet
vector_index_stats_limit153121_bge_base.json
query_understanding_examples_mesh.csv
```

中文正式报告可保留中文标题，但必须包含范围或阶段后缀，避免出现无法区分范围的 `最终版`、`最新版`、`new`、`copy`。

## 状态与归档

- 正式可复用产物放在 `artifacts/` 或 `reports/formal/`。
- 调试、基线和已被替代实验放在 `archive/experiments/`。
- 可重新生成的缓存放在 `runtime/cache/`。
- 发布压缩包放在 `releases/<topic>/`，不放在项目根目录或通用输出目录。
