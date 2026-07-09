"""
年报 RAG 问答后端服务。

本文件只负责 FastAPI 后端接口，不直接写前端页面代码。
前端页面放在 frontend/index.html，/ 路由只负责把该 HTML 文件返回给浏览器。

运行示例:
    python src/api_server.py
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"

try:
    from . import rag_qa
except ImportError:  # 兼容 python src/api_server.py 这种直接运行方式。
    import rag_qa


# /api/qa 的请求体模型，定义问题、过滤条件、检索参数和展示参数。
class QARequest(BaseModel):
    """前端调用 /api/qa 时传入的请求参数。"""

    # 用户问题，必填。
    question: str = Field(..., min_length=1)

    # 可选过滤条件；不传时会尝试从问题中自动识别公司、股票代码和年份。
    stock_code: str | None = None
    year: str | None = None
    years: list[str] | None = None

    # 最终给 LLM 的上下文条数。
    top_k: int = Field(default=rag_qa.DEFAULT_TOP_K, ge=1, le=20)

    # 各检索阶段的候选数量和阈值，前端可以开放成调参面板。
    vector_top_k: int = Field(default=rag_qa.DEFAULT_VECTOR_TOP_K, ge=1, le=200)
    bm25_top_k: int = Field(default=rag_qa.DEFAULT_BM25_TOP_K, ge=1, le=200)
    rerank_candidates: int = Field(default=rag_qa.DEFAULT_RERANK_CANDIDATES, ge=1, le=200)
    rrf_k: int = Field(default=rag_qa.DEFAULT_RRF_K, ge=1, le=200)
    rerank_threshold: float = Field(default=rag_qa.DEFAULT_RERANK_THRESHOLD, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=rag_qa.DEFAULT_MAX_CONTEXT_CHARS, ge=500, le=30000)

    # 模型名称默认沿用 rag_qa.py 里的配置。
    embedding_model: str = rag_qa.DEFAULT_EMBEDDING_MODEL
    rerank_model: str = rag_qa.DEFAULT_RERANK_MODEL
    llm_model: str = rag_qa.DEFAULT_LLM_MODEL

    # include_content=True 时返回完整 chunk 内容；默认只返回预览，避免响应太大。
    include_content: bool = False
    content_preview_chars: int = Field(default=300, ge=0, le=3000)


# RAG 服务层：启动时加载索引，处理健康检查、配置查询和问答请求。
class RagService:
    """
    RAG 服务封装。

    这里在服务启动时只加载一次 FAISS 索引和 metadata，避免每次请求都重复读磁盘。
    具体的检索、rerank、LLM 调用仍然复用 rag_qa.py 中已有函数。
    """

    def __init__(
        self,
        index_path: str = rag_qa.DEFAULT_INDEX_PATH,
        metadata_path: str = rag_qa.DEFAULT_METADATA_PATH,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index, self.metadata_items = rag_qa.load_index_and_metadata(index_path, metadata_path)

    def health(self) -> dict[str, Any]:
        """返回服务健康状态，以及索引和 metadata 的基本数量。"""

        return {
            "status": "ok",
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
            "index_vectors": int(self.index.ntotal),
            "metadata_items": len(self.metadata_items),
        }

    def config(self) -> dict[str, Any]:
        """返回当前后端默认参数，方便前端初始化配置面板。"""

        return {
            "embedding_model": rag_qa.DEFAULT_EMBEDDING_MODEL,
            "rerank_model": rag_qa.DEFAULT_RERANK_MODEL,
            "llm_model": rag_qa.DEFAULT_LLM_MODEL,
            "top_k": rag_qa.DEFAULT_TOP_K,
            "vector_top_k": rag_qa.DEFAULT_VECTOR_TOP_K,
            "bm25_top_k": rag_qa.DEFAULT_BM25_TOP_K,
            "rerank_candidates": rag_qa.DEFAULT_RERANK_CANDIDATES,
            "rrf_k": rag_qa.DEFAULT_RRF_K,
            "rerank_threshold": rag_qa.DEFAULT_RERANK_THRESHOLD,
            "max_context_chars": rag_qa.DEFAULT_MAX_CONTEXT_CHARS,
        }

    def companies(self) -> list[dict[str, Any]]:
        """从 metadata 中聚合公司、股票代码、年份，用于前端筛选下拉框。"""

        grouped: dict[tuple[str, str], set[str]] = {}
        for item in self.metadata_items:
            meta = item.get("meta", {})
            company = str(meta.get("company_name", "")).strip()
            code = str(meta.get("stock_code", "")).strip()
            year = str(meta.get("year", "")).strip()
            if not company and not code:
                continue
            grouped.setdefault((company, code), set())
            if year:
                grouped[(company, code)].add(year)

        return [
            {
                "company_name": company,
                "stock_code": code,
                "years": sorted(years),
            }
            for (company, code), years in sorted(grouped.items(), key=lambda item: item[0])
        ]

    def answer(self, request: QARequest) -> dict[str, Any]:
        """
        执行完整 RAG 问答流程，并返回答案和每一层候选。

        流程:
            1. 读取 API Key
            2. 解析显式/隐式过滤条件
            3. query embedding
            4. FAISS 向量召回
            5. BM25 关键词召回
            6. RRF 融合
            7. rerank 精排
            8. 构造上下文并调用 LLM
            9. 返回 answer、references、pipeline、usage
        """

        api_key = rag_qa.get_api_key()

        # 合并用户显式传入的 stock_code/year 和从问题中自动识别出的过滤条件。
        resolved_filters = rag_qa.resolve_filters(
            question=request.question,
            metadata_items=self.metadata_items,
            stock_code=request.stock_code,
            year=request.year,
            years=request.years,
        )

        # 1. 向量召回：问题先转 embedding，再用 FAISS 搜索相似 chunk。
        query_vector = rag_qa.embed_query(request.question, request.embedding_model, api_key)
        vector_hits = rag_qa.vector_recall(
            index=self.index,
            metadata_items=self.metadata_items,
            query_vector=query_vector,
            vector_top_k=request.vector_top_k,
            stock_code=resolved_filters["stock_code"],
            year=resolved_filters["year"],
            years=resolved_filters["years"],
        )

        # 2. BM25 召回：基于 jieba 分词和关键词匹配找候选。
        bm25_hits = rag_qa.bm25_recall(
            metadata_items=self.metadata_items,
            question=request.question,
            top_n=request.bm25_top_k,
            stock_code=resolved_filters["stock_code"],
            year=resolved_filters["year"],
            years=resolved_filters["years"],
        )

        # 3. RRF 融合：把向量召回和 BM25 召回的排序合并成一个候选池。
        rrf_hits = rag_qa.rrf_fusion(vector_hits, bm25_hits, rrf_k=request.rrf_k)

        # 4. rerank 精排：只把 RRF 排名前 rerank_candidates 的候选送入精排模型。
        rerank_input = rrf_hits[: request.rerank_candidates]
        final_candidates = rag_qa.rerank_candidates(
            question=request.question,
            candidates=rerank_input,
            model=request.rerank_model,
            api_key=api_key,
            threshold=request.rerank_threshold,
            top_k=request.top_k,
        )

        # 5. 构造最终上下文并调用 LLM；references 就是最终答案引用来源。
        context, references = rag_qa.build_context(final_candidates, request.max_context_chars)
        if context:
            answer = rag_qa.call_llm(request.question, context, request.llm_model, api_key)
        else:
            answer = "No answer can be determined from the provided documents."

        filters = {
            "explicit_stock_code": request.stock_code,
            "explicit_year": request.year,
            "explicit_years": request.years,
            "inferred_stock_code": resolved_filters["inferred_stock_code"],
            "inferred_year": resolved_filters["inferred_year"],
            "inferred_years": resolved_filters["inferred_years"],
            "matched_company": resolved_filters["matched_company"],
            "resolved_stock_code": resolved_filters["stock_code"],
            "resolved_year": resolved_filters["year"],
            "resolved_years": resolved_filters["years"],
        }

        # pipeline 给前端展示每个阶段的候选列表和分数。
        pipeline = {
            "vector_recall": [
                format_candidate(hit, "vector_recall", index + 1, request)
                for index, hit in enumerate(vector_hits)
            ],
            "bm25_recall": [
                format_candidate(hit, "bm25_recall", index + 1, request)
                for index, hit in enumerate(bm25_hits)
            ],
            "rrf_fusion": [
                format_candidate(hit, "rrf_fusion", index + 1, request)
                for index, hit in enumerate(rrf_hits)
            ],
            "rerank_input": [
                format_candidate(hit, "rerank_input", index + 1, request)
                for index, hit in enumerate(rerank_input)
            ],
            "rerank": [
                format_candidate(hit, "rerank", index + 1, request)
                for index, hit in enumerate(final_candidates)
            ],
            "final_context": references,
        }

        return {
            "question": request.question,
            "answer": answer,
            "filters": filters,
            "references": references,
            "pipeline": pipeline,
            "usage": {
                "vector_count": len(vector_hits),
                "bm25_count": len(bm25_hits),
                "rrf_count": len(rrf_hits),
                "rerank_input_count": len(rerank_input),
                "rerank_output_count": len(final_candidates),
                "final_reference_count": len(references),
                "context_chars": len(context),
            },
        }


# 把内部候选结构转换成前端更容易展示的统一结构。
def format_candidate(candidate: dict[str, Any], stage: str, rank: int, request: QARequest) -> dict[str, Any]:
    """把 rag_qa.py 内部候选结构转换成前端更容易展示的统一结构。"""

    item = candidate.get("item", {})
    meta = item.get("meta", {})
    content = item.get("content", "") or ""
    payload = {
        "rank": rank,
        "chunk_id": candidate.get("chunk_id"),
        "vector_id": candidate.get("vector_id"),
        "source_stage": stage,
        "sources": candidate.get("sources", []),
        "scores": {
            "vector_rank": candidate.get("vector_rank"),
            "vector_score": candidate.get("vector_score"),
            "bm25_rank": candidate.get("bm25_rank"),
            "bm25_score": candidate.get("bm25_score"),
            "rrf_score": candidate.get("rrf_score"),
            "rerank_rank": candidate.get("rerank_rank"),
            "rerank_score": candidate.get("rerank_score"),
        },
        "meta": meta,
        "content_preview": content[: request.content_preview_chars],
    }
    if request.include_content:
        payload["content"] = content
    return payload


# FastAPI 应用对象。前端或其他服务实际调用的就是这个 app。
app = FastAPI(title="Annual Report RAG API", version="1.0.0")

# 允许前端跨域调用。开发阶段先放开，后续上线时可以改成指定前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务启动时立即加载索引，后续请求复用同一个 service。
service = RagService()


# 页面入口：返回外部 frontend/index.html，前端代码不写在 server 文件里。
@app.get("/", response_class=FileResponse)
async def frontend() -> FileResponse:
    """返回外部前端页面 frontend/index.html。"""

    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail=f"Frontend file not found: {FRONTEND_INDEX}")
    return FileResponse(FRONTEND_INDEX)

# 健康检查接口：确认后端启动、索引和 metadata 已加载。
@app.get("/api/health")
async def health() -> dict[str, Any]:
    """健康检查接口。"""

    return service.health()


# 默认配置接口：返回模型名、top_k、召回数量等前端初始化参数。
@app.get("/api/config")
async def config() -> dict[str, Any]:
    """默认配置接口。"""

    return service.config()


# 公司列表接口：返回已入库公司、股票代码和年份，用于前端筛选。
@app.get("/api/companies")
async def companies() -> list[dict[str, Any]]:
    """公司/股票代码/年份列表接口。"""

    return service.companies()


# 核心问答接口：执行完整 RAG 流程，并返回最终答案和各阶段候选。
@app.post("/api/qa")
async def qa(request: QARequest) -> dict[str, Any]:
    """核心问答接口：返回最终答案和各检索阶段候选信息。"""

    try:
        # RAG 流程里有同步 SDK 调用，放到线程池里避免阻塞 FastAPI 事件循环。
        return await run_in_threadpool(service.answer, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# 本地启动入口：配置导入路径并启动 Uvicorn 服务。
def main() -> None:
    """本地启动入口，支持直接运行 python src/api_server.py。"""

    import sys
    from pathlib import Path

    import uvicorn

    # 直接运行 src/api_server.py 时，Python 默认只把 src 加到路径里。
    # 这里补上项目根目录，确保 uvicorn 能导入 src.api_server:app。
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 当前 Windows 环境下 reload 子进程可能触发 WinError 5，所以默认关闭。
    uvicorn.run(
        "src.api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        app_dir=str(project_root),
    )


if __name__ == "__main__":
    main()
