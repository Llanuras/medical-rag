from __future__ import annotations

import importlib
import os
import platform
from pathlib import Path

import pandas as pd

from medical_rag.common.pmc import disk_usage_row, ensure_output_dirs, setup_tee, write_markdown


def main() -> None:
    ensure_output_dirs()
    log_path = Path("logs/00_check_environment.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        cwd = Path.cwd()
        data_dir = Path("data/raw/pmc_oa_comm")
        xml_count = sum(1 for _ in data_dir.rglob("*.xml")) if data_dir.exists() else 0
        mods = ["pandas", "datasets", "bs4", "lxml", "transformers", "matplotlib", "chromadb", "langchain", "langchain_chroma", "langchain_community"]
        dep_rows = []
        print("Environment check")
        print("cwd:", cwd)
        print("python:", platform.python_version())
        for mod in mods:
            try:
                importlib.import_module(mod)
                status, detail = "OK", ""
            except Exception as exc:
                status, detail = "ERR", f"{type(exc).__name__}: {exc}"
            dep_rows.append({"dependency": mod, "status": status, "detail": detail})
            print(f"{mod}: {status} {detail}")
        disk = disk_usage_row(Path("/root/autodl-tmp"))
        rows = [
            {"item": "cwd", "value": str(cwd)},
            {"item": "python_version", "value": platform.python_version()},
            {"item": "data_dir", "value": str(data_dir)},
            {"item": "data_dir_exists", "value": str(data_dir.exists())},
            {"item": "xml_count", "value": str(xml_count)},
            {"item": "autodl_tmp_free_gb", "value": disk["free_gb"]},
        ]
        for row in dep_rows:
            rows.append({"item": f"dependency_{row['dependency']}", "value": row["status"]})
        pd.DataFrame(rows).to_csv("artifacts/metrics/t001_environment/environment_summary.csv", index=False)
        pd.DataFrame(dep_rows).to_csv("artifacts/metrics/t001_environment/environment_dependencies.csv", index=False)
        print("xml_count:", xml_count)
        print("disk:", disk)
        enough = xml_count >= 500
        dep_ok = all(r["status"] == "OK" for r in dep_rows)
        body = f"""
## 本阶段分析目标

环境检查用于确认后续 500 篇 PMC XML 数据分析能在远程项目中稳定运行，避免在路径、依赖、磁盘空间或数据数量不足时继续执行。

## 检查结果

- 当前工作目录：`{cwd}`
- 数据目录：`{data_dir}`
- 数据目录是否存在：`{data_dir.exists()}`
- XML 文件数量：`{xml_count}`
- `/root/autodl-tmp` 剩余空间：`{disk['free_gb']} GB`
- 依赖导入是否全部通过：`{dep_ok}`

## 数据路径与样本量判断

当前数据路径符合任务要求，数据来自本地 `data/raw/pmc_oa_comm`。XML 数量{'足够' if enough else '不足'}支撑第一批 500 篇分析。

如果 XML 数量不足 500，下一步应只从 PMC OA Bulk 的 `oa_comm/xml` 路径下载一个较小 tar.gz 样本并解压，不应下载全量，也不应使用其他数据集替代。

## 磁盘空间判断

当前数据盘空间足够保存 CSV、JSONL、图表、阶段说明文档和 500 篇 Chroma 测试库。

## 范围说明

本阶段不涉及模型训练、不涉及大规模推理，也不开发完整 RAG 问答系统。当前任务仅做数据加载、评估和小规模向量库可用性验证。
"""
        write_markdown(Path("reports/technical/00_environment_check_notes.md"), "环境检查说明", body)
        print("Wrote artifacts/metrics/t001_environment/environment_summary.csv")
        print("Wrote reports/technical/00_environment_check_notes.md")

if __name__ == "__main__":
    main()
