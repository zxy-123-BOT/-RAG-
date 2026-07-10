# 年报 RAG 智能问答系统

这是一个面向上市公司年报的本地 RAG（Retrieval-Augmented Generation，检索增强生成）项目。项目可以从年报 PDF 中解析正文和表格，构建 chunk，生成向量索引，并通过 FastAPI 后端和前端页面完成财报问答与检索链路可视化。

当前系统支持：

- 年报 PDF 批量解析
- 表格抽取与 Markdown 化
- 文本 chunk 和表格 chunk 构建
- DashScope / 百炼 embedding
- FAISS 向量检索
- BM25 关键词检索
- RRF 多路召回融合
- DashScope rerank 精排
- qwen-plus 基于引用上下文生成答案
- 前端展示最终答案和每一阶段候选结果

## 项目结构

```text
.
├── frontend/
│   └── index.html              # 前端页面，展示问答结果和检索流程候选
├── src/
│   ├── parse_reports.py        # PDF 年报解析为结构化 JSON
│   ├── chunk_reports.py        # parsed_json 切分为 RAG chunks
│   ├── build_index.py          # 调用 embedding，构建 FAISS 索引
│   ├── rag_qa.py               # 命令行 RAG 问答主流程
│   └── api_server.py           # FastAPI 后端服务
└── data/
    ├── raw_pdf/                # 原始年报 PDF
    ├── parsed_json/            # PDF 解析后的结构化 JSON
    ├── chunks/                 # 切片结果，含 all_chunks.json
    └── index/                  # FAISS 索引、metadata、embedding 缓存
```

## RAG 流程

```text
PDF 年报
  ↓ parse_reports.py
结构化 parsed_json
  ↓ chunk_reports.py
RAG chunks
  ↓ build_index.py
embedding + FAISS index
  ↓ rag_qa.py / api_server.py
向量召回 + BM25 召回 + RRF 融合 + rerank 精排
  ↓
LLM 生成最终答案
```

## 环境要求

建议使用 Python 3.10+。

主要依赖：

```text
fastapi
uvicorn
dashscope
faiss-cpu
numpy
jieba
rank-bm25
pymupdf
pdfplumber
tqdm
```

安装示例：

```bash
pip install fastapi uvicorn dashscope faiss-cpu numpy jieba rank-bm25 pymupdf pdfplumber tqdm
```

如果你使用 Anaconda，也可以在虚拟环境中安装：

```bash
conda create -n rag-report python=3.10
conda activate rag-report
pip install fastapi uvicorn dashscope faiss-cpu numpy jieba rank-bm25 pymupdf pdfplumber tqdm
```

## 配置 API Key

本项目使用阿里云 DashScope / 百炼模型服务，需要配置环境变量：

```text
DASHSCOPE_API_KEY
```

PowerShell 临时设置：

```powershell
$env:DASHSCOPE_API_KEY="你的 API Key"
```

Windows 当前用户永久设置：

```powershell
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "你的 API Key", "User")
```

永久设置后需要重新打开终端，再检查：

```powershell
echo $env:DASHSCOPE_API_KEY
```

## 数据准备

把年报 PDF 放到：

```text
data/raw_pdf/
```

示例：

```text
data/raw_pdf/良品铺子：良品铺子股份有限公司2020年年度报告.PDF
data/raw_pdf/中芯国际：中芯国际2024年年度报告.pdf
```

注意：PDF 原文、embedding 缓存、FAISS 索引文件通常较大，不建议直接提交到 GitHub。推荐在本地按下面流程重新生成。

## 1. 解析 PDF

```bash
python src/parse_reports.py
```

只解析前 1 个 PDF 做测试：

```bash
python src/parse_reports.py --limit 1
```

输出目录：

```text
data/parsed_json/
```

## 2. 构建 chunks

```bash
python src/chunk_reports.py
```

可指定每个文本 chunk 的最大长度：

```bash
python src/chunk_reports.py --max-text-chars 1000
```

输出目录：

```text
data/chunks/
```

其中：

```text
data/chunks/all_chunks.json
```

会作为后续构建向量索引的输入。

## 3. 构建 FAISS 索引

```bash
python src/build_index.py
```

只用前 20 条 chunk 测试：

```bash
python src/build_index.py --limit 20
```

删除旧 embedding 缓存并重建：

```bash
python src/build_index.py --rebuild
```

输出目录：

```text
data/index/
```

主要文件：

```text
faiss.index             # FAISS 向量索引
metadata.json           # vector_id 对应的 chunk 元数据
index_config.json       # 索引构建配置
embedding_cache.jsonl   # embedding 缓存，便于断点续建
failed_chunks.json      # embedding 失败记录
```

## 4. 命令行问答

```bash
python src/rag_qa.py --question "良品铺子2020年营业收入是多少？"
```

指定股票代码和年份：

```bash
python src/rag_qa.py --question "营业收入是多少？" --stock-code 603719 --year 2020
```

保存 debug 结果：

```bash
python src/rag_qa.py --question "良品铺子2020年营业收入是多少？" --debug-output data/index/last_debug.json
```

## 5. 启动后端服务

```bash
python src/api_server.py
```

启动后浏览器访问：

```text
http://127.0.0.1:8000/
```

后端接口：

```text
GET  /                返回 frontend/index.html
GET  /api/health      健康检查
GET  /api/config      默认参数
GET  /api/companies   已入库公司、股票代码、年份
POST /api/qa          核心问答接口
```

## API 示例

请求：

```http
POST /api/qa
Content-Type: application/json
```

```json
{
  "question": "良品铺子2020年营业收入是多少？",
  "stock_code": null,
  "year": null,
  "top_k": 4,
  "vector_top_k": 20,
  "bm25_top_k": 20,
  "rerank_candidates": 20,
  "include_content": true,
  "content_preview_chars": 1200
}
```

响应会包含：

```json
{
  "question": "...",
  "answer": "...",
  "filters": {},
  "references": [],
  "pipeline": {
    "vector_recall": [],
    "bm25_recall": [],
    "rrf_fusion": [],
    "rerank_input": [],
    "rerank": [],
    "final_context": []
  },
  "usage": {}
}
```

前端展示的检索流程候选就是来自 `pipeline` 字段：

```text
向量召回        pipeline.vector_recall
BM25 召回       pipeline.bm25_recall
RRF 融合        pipeline.rrf_fusion
Rerank 输入     pipeline.rerank_input
Rerank 精排     pipeline.rerank
```

## 年份识别说明

系统会从问题中自动识别年份条件。

单一年份：

```text
良品铺子2020年营业收入是多少？
```

识别为：

```json
{
  "year": "2020",
  "years": null
}
```

年份范围：

```text
良品铺子2019到2021营业收入变化趋势是怎样的？
```

识别为：

```json
{
  "year": null,
  "years": ["2019", "2020", "2021"]
}
```

多个年份：

```text
良品铺子2019年、2020年、2021年营业收入分别是多少？
```

识别为：

```json
{
  "year": null,
  "years": ["2019", "2020", "2021"]
}
```

## 主要参数说明

| 参数 | 含义 |
| --- | --- |
| `top_k` | 最终送给 LLM 的上下文数量 |
| `vector_top_k` | FAISS 向量召回数量 |
| `bm25_top_k` | BM25 关键词召回数量 |
| `rerank_candidates` | RRF 后送入 rerank 模型的候选数量 |
| `rrf_k` | RRF 融合平滑参数 |
| `rerank_threshold` | rerank 分数过滤阈值 |
| `max_context_chars` | 最终上下文最大字符数 |

趋势类、多年份问题建议适当调大：

```text
top_k
rerank_candidates
```

否则可能某些年份的证据没有进入最终上下文。


## 常见问题

### 1. `DASHSCOPE_API_KEY` 没生效

如果使用永久环境变量设置，需要重新打开终端。

当前 PowerShell 临时生效：

```powershell
$env:DASHSCOPE_API_KEY="你的 API Key"
```

### 2. 端口 8000 被占用

说明已有服务正在运行。可以关闭旧服务，或修改 `src/api_server.py` 中的端口。

### 3. `jieba` 出现 `pkg_resources is deprecated`

这是依赖库警告，通常不影响运行。

### 4. 知识库外问题回答不准

这是 RAG 系统常见问题。建议在后续版本中增加：

- 公司识别失败保护
- 低 rerank 分数拒答
- 引用证据强校验

## License

你可以根据自己的需求选择开源协议，例如 MIT License。
