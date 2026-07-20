# Token 长度分析与 embedding 输入限制说明

## 1. 本阶段分析目标

Token 长度决定文本是否能直接进入 embedding 模型。RAG 构建前需要统计长度分布，避免超出模型输入限制导致截断或信息丢失。

## 2. 为什么使用 embedding 模型对应 tokenizer

字符数、词数和 token 数并不等价。Embedding 模型通常按 tokenizer 的 token 序列处理输入，因此本阶段使用 `sentence-transformers/all-MiniLM-L6-v2` 对应 tokenizer，并按 512 tokens 作为关键参考上限。

## 3. 分析对象选择

本阶段同时分析 `title+abstract` 和 `title+abstract+body`。初始 RAG 优先考虑 `title+abstract`，因为摘要信息密度高、长度可控；全文适合后续更复杂的章节切分和检索。

## 4. 长度分布指标解释

mean 和 median 描述整体长度水平；p95 和 p99 用于判断绝大多数样本是否能直接入库；max 反映长尾风险。如果 p95 接近或超过 512，则需要更积极的切分策略。

## 5. 可选策略

- 整体不切分：优点是上下文完整、实现简单；缺点是超长文本可能超过 embedding 限制。
- 只对长尾样本切分：优点是保留大多数摘要完整性；缺点是实现略复杂。
- 全量统一滑动窗口切分：优点是格式统一；缺点是短摘要会被不必要切分。
- 全文先章节切分再窗口切分：优点是适合结构清晰论文；缺点是依赖章节结构，XML 不统一时实现复杂。

## 6. 当前 500 篇真实数据结果

- title+abstract p95：`530.00`
- title+abstract p99：`614.16`
- title+abstract 超过 512 tokens 比例：`6.0000%`
- full text p95：`15001.25`
- full text p99：`22022.38`
- full text 超过 512 tokens 比例：`98.0000%`

## 7. 初步策略选择

title+abstract 存在明显超长比例，建议使用长尾滑动窗口或统一切分。

对于全文 RAG，当前结果显示全文通常远长于摘要，因此正式使用 body 时应采用章节切分 + recursive split，而不是整体 embedding。
