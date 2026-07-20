# 医学 RAG 本地大模型环境准备与验证报告

日期：2026-06-23

## 1. 项目目标

本阶段目标是为后续医学专业知识生成型 LLM / RAG 系统开发完成基础环境准备，并验证关键技术链路是否可用。

当前阶段属于准备工作，不进行完整 RAG 系统开发，重点验证以下内容：

- 本地大语言模型能否正常运行
- GPU 与 PyTorch 环境是否可用
- Chroma 向量数据库是否能够写入和检索
- PMC OA 医学文献 XML 数据源是否能够引入和解析
- LangChain 是否能够调用本地 Ollama 模型

## 2. 技术方案

当前采用的技术路线如下：

- 云服务器：AutoDL
- 操作系统：Ubuntu 22.04
- GPU：NVIDIA GeForce RTX 4090 24GB
- CUDA / PyTorch：PyTorch 2.5.1+cu124
- 本地模型运行框架：Ollama
- 本地模型：Qwen3-8B Q4 量化版本
- RAG 框架：LangChain
- 向量数据库：Chroma
- Embedding 模型：sentence-transformers/all-MiniLM-L6-v2
- 医学文献数据源：NCBI PMC OA `oa_comm/xml` 小样本

项目目录：

```text
/root/autodl-tmp/medical_rag
```

Ollama 模型目录：

```text
/root/autodl-tmp/medical_rag/artifacts/models/ollama
```

## 3. 项目结构

当前项目结构如下：

```text
medical_rag/
├── .venv/
├── archive/experiments/indexes/chroma_test_db/
├── data/
│   └── pmc_oa_comm/
│       ├── oa_comm_xml.PMC000xxxxxx.baseline.2026-06-18.tar.gz
│       └── PMC000xxxxxx/
├── artifacts/models/huggingface/
├── artifacts/models/ollama/
└── scripts/
    ├── check_gpu.py
    ├── parse_pmc_sample.py
    ├── test_chroma.py
    └── test_ollama_langchain.py
```

其中：

- `.venv/`：Python 虚拟环境
- `artifacts/models/ollama/`：Ollama 本地模型存储目录
- `artifacts/models/huggingface/`：HuggingFace 模型缓存目录
- `archive/experiments/indexes/chroma_test_db/`：Chroma 测试向量库目录
- `data/raw/pmc_oa_comm/`：PMC OA 医学文献 XML 小样本数据
- `scripts/`：本阶段环境验证脚本

## 4. 已完成工作

### 4.1 Python 与 PyTorch 环境

已创建 Python 虚拟环境：

```text
/root/autodl-tmp/medical_rag/.venv
```

当前 Python 版本：

```text
Python 3.12.3
```

当前 PyTorch 版本：

```text
2.5.1+cu124
```

在 RTX 4090 有卡模式下，已验证 PyTorch 可以识别 GPU。

验证结果包括：

```text
CUDA available: True
GPU name: NVIDIA GeForce RTX 4090
GPU memory GB: 23.516...
```

### 4.2 Ollama 本地模型

已安装 Ollama，并将模型存放在数据盘目录，避免占用系统盘空间：

```text
/root/autodl-tmp/medical_rag/artifacts/models/ollama
```

已下载并验证本地模型：

```text
qwen3:8b
```

通过 `ollama run qwen3:8b` 已确认模型可以正常进行中文问答。

### 4.3 Chroma 向量数据库验证

已编写并运行 Chroma 测试脚本：

```text
scripts/test_chroma.py
```

测试流程：

1. 构造少量测试文档
2. 使用 HuggingFace Embedding 模型生成文本向量
3. 写入 Chroma 本地向量数据库
4. 执行相似度检索

验证输出中成功返回：

```text
Search results:

[1] Chroma is a vector database for storing document embeddings.
```

该结果说明 Chroma 的向量写入与检索流程已经打通。

![Chroma 向量数据库写入与检索测试结果](assets/chroma_test_result.png)

### 4.4 PMC OA 医学文献数据源引入

已从 NCBI PMC OA `oa_comm/xml` 数据源下载小样本数据：

```text
oa_comm_xml.PMC000xxxxxx.baseline.2026-06-18.tar.gz
```

解压后得到约 3028 个 XML 文件。

已编写并运行 XML 解析脚本：

```text
scripts/parse_pmc_sample.py
```

脚本可以成功解析文献中的以下内容：

- Title
- Abstract preview
- Body preview



### 4.5 LangChain 调用 Ollama 验证

已编写测试脚本：

```text
scripts/test_ollama_langchain.py
```

该脚本通过 LangChain 调用本地 Ollama 中的 `qwen3:8b` 模型，并成功生成中文回答。

 LangChain 与本地 Ollama 模型之间的调用链路已经打通，后续可以在此基础上接入 Retriever，实现基础 RAG 问答流程。

### 4.6 本地模型中文问答验证

已通过 Ollama 直接与本地 Qwen3-8B 模型进行中文问答测试。模型能够正常加载并返回中文回答。

![本地 Qwen3-8B 模型中文问答测试结果](assets/ollama_chat_result.png)

## 5. 当前验证结果汇总

| 验证项 | 当前状态 | 说明 |
| --- | --- | --- |
| AutoDL GPU 环境 | 已通过 | 有卡模式下识别 RTX 4090 |
| PyTorch CUDA | 已通过 | `CUDA available: True` |
| Ollama 模型 | 已通过 | `ollama list` 可看到 `qwen3:8b` |
| Chroma 向量库 | 已通过 | 可写入文档并完成相似度检索 |
| PMC XML 数据源 | 已通过 | 已引入并解析 PMC OA XML 小样本 |
| Ollama 本地问答 | 已通过 | `qwen3:8b` 可正常生成中文回答 |
| LangChain 调用 Ollama | 已通过 | 可通过 LangChain 调用本地模型 |

## 6. 当前结论

目前已完成医学 RAG 项目前期环境准备与基础验证工作。

当前环境已经具备以下能力：

- 使用 RTX 4090 运行本地大语言模型
- 通过 Ollama 管理和调用 Qwen3-8B 模型
- 使用 LangChain 调用本地模型
- 使用 Chroma 存储和检索文本向量
- 引入并解析 PMC OA 医学文献 XML 数据



