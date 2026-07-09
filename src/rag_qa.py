"""
年报 RAG 命令行问答主流程。

本文件负责加载 FAISS 索引和 metadata，执行问题解析、向量召回、BM25 召回、RRF 融合、rerank 精排和 LLM 生成。
它既可以作为命令行脚本运行，也会被 api_server.py 复用。

运行示例:
    python src/rag_qa.py --question "良品铺子2020年营业收入是多少？"
    python src/rag_qa.py --question "良品铺子2019到2021年营业收入变化趋势是怎样的？"
"""

import argparse
import json
import os
import re
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

try:
    from dashscope import Generation, TextEmbedding, TextReRank
except ImportError as exc:
    raise RuntimeError("缺少 dashscope SDK, 请先运行: python -m pip install dashscope") from exc


DEFAULT_INDEX_PATH = "data/index/faiss.index"
DEFAULT_METADATA_PATH = "data/index/metadata.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_RERANK_MODEL = "gte-rerank-v2"
DEFAULT_LLM_MODEL = "qwen-plus"
DEFAULT_TOP_K = 4
DEFAULT_VECTOR_TOP_K = 20
DEFAULT_BM25_TOP_K = 20
DEFAULT_RERANK_CANDIDATES = 20#RRF
DEFAULT_RRF_K = 60 #平滑参数
DEFAULT_RERANK_THRESHOLD = 0.25
DEFAULT_MAX_CONTEXT_CHARS = 8000


SYSTEM_PROMPT = """你是一个金融财报问答助手。
你只能根据给定的上下文回答问题，不得编造。
如果上下文中没有足够信息，请回答“不知道”或“根据已提供资料无法判断”。
回答必须简洁、准确，直接给出结论。
回答必须是完整句子，包含用户问题中的主体和指标，不要只输出一个数字。
只回答用户问题本身，不要展开解释检索过程、不要说明“多次出现”、不要做不必要的单位换算说明。
如果需要补充依据，最多补充一句。
涉及数字、年份、公司、财务指标时必须严格依据上下文。
每个关键结论后必须标注引用编号，例如【1】、【2】。
不要引用没有使用过的上下文。"""


# 读取 UTF-8 JSON 文件。
def read_json(path):
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 写入格式化 UTF-8 JSON 文件。
def write_json(path, data):
    """Write a UTF-8 JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 从环境变量 DASHSCOPE_API_KEY 读取百炼 API Key。
def get_api_key():
    """Read DashScope API key from environment."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY。")
    return api_key


# 从 DashScope embedding 响应中提取向量，兼容不同 SDK 返回格式。
def extract_embeddings_from_response(response):
    """Extract embeddings from DashScope response."""
    if hasattr(response, "status_code") and response.status_code != 200:
        message = getattr(response, "message", "") or str(response)
        raise RuntimeError(f"DashScope embedding failed: {message}")
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    if output is None:
        raise RuntimeError(f"DashScope response missing output: {response}")
    embeddings = output.get("embeddings") if isinstance(output, dict) else getattr(output, "embeddings", None)
    if not embeddings:
        raise RuntimeError(f"DashScope response missing embeddings: {response}")

    result = []
    for item in embeddings:
        result.append(item["embedding"] if isinstance(item, dict) else item.embedding)
    return result


# 把用户问题转换成 query embedding，并做 L2 归一化。
def embed_query(question, model, api_key):
    """Embed query text and L2-normalize it for IndexFlatIP cosine search."""
    response = TextEmbedding.call(model=model, input=[question], api_key=api_key)
    embedding = extract_embeddings_from_response(response)[0]
    vector = np.asarray([embedding], dtype="float32")
    faiss.normalize_L2(vector)
    return vector


# 加载 FAISS 索引和与 vector_id 对齐的 metadata。
def load_index_and_metadata(index_path, metadata_path):
    """Load FAISS index and vector-id aligned metadata items."""
    index = faiss.read_index(str(index_path))
    metadata = read_json(metadata_path).get("items", [])
    if index.ntotal != len(metadata):
        raise RuntimeError(f"index.ntotal({index.ntotal}) != metadata items({len(metadata)})")
    return index, metadata


# 把 2019 到 2021 这种年份范围展开成逐年列表。
def expand_year_range(start_year, end_year, max_span=10):
    """Expand a year range such as 2019-2021 into ["2019", "2020", "2021"]."""
    start = int(start_year)
    end = int(end_year)
    if start > end:
        start, end = end, start
    if end - start > max_span:
        return []
    return [str(year) for year in range(start, end + 1)]


# 从问题中识别单一年份、多个年份或年份范围。
def extract_years_from_question(question):
    """
    Extract year filters from a question.

    Returns:
        (year, years)

    year is used for a single-year question, while years is used for range or
    multi-year questions. This avoids incorrectly filtering "2019到2021年..."
    down to only 2019.
    """
    range_match = re.search(
        r"(20\d{2})\s*年?\s*(?:到|至|~|～|-|—|－)\s*(20\d{2})\s*年?",
        question,
    )
    if range_match:
        years = expand_year_range(range_match.group(1), range_match.group(2))
        if years:
            return None, years

    years = sorted(set(re.findall(r"20\d{2}", question)))
    if len(years) > 1:
        return None, years
    if len(years) == 1:
        return years[0], None
    return None, None


# 把可选年份列表规范成字符串集合。
def normalize_years(years):
    """Normalize optional year-list input to a string set."""
    if not years:
        return None
    normalized = {str(year).strip() for year in years if str(year).strip()}
    return normalized or None


# 判断某个 metadata item 是否符合股票代码和年份过滤条件。
def item_matches_filter(item, stock_code=None, year=None, years=None):
    """Return True if metadata item matches optional filters."""
    meta = item.get("meta", {})
    if stock_code and str(meta.get("stock_code", "")) != str(stock_code):
        return False
    allowed_years = normalize_years(years)
    if allowed_years is not None and str(meta.get("year", "")) not in allowed_years:
        return False
    if year and str(meta.get("year", "")) != str(year):
        return False
    return True


# 从用户问题中自动识别股票代码、公司名和年份条件。
def infer_filters_from_question(question, metadata_items):
    """
    Infer stock_code and year filters from a user question.

    The mapping is built from metadata.json instead of hard-coded company names.
    If nothing is detected, both values stay None and retrieval remains unfiltered.
    """
    inferred = {
        "stock_code": None,
        "year": None,
        "years": None,
        "matched_company": None,
    }

    inferred["year"], inferred["years"] = extract_years_from_question(question)

    stock_match = re.search(r"\b[03668]\d{5}\b", question)
    if stock_match:
        inferred["stock_code"] = stock_match.group(0)
        return inferred

    company_to_code = {}
    for item in metadata_items:
        meta = item.get("meta", {})
        company = str(meta.get("company_name", "")).strip()
        code = str(meta.get("stock_code", "")).strip()
        if company and code:
            company_to_code[company] = code

    # Prefer longer company names first when one name contains another.
    for company in sorted(company_to_code, key=len, reverse=True):
        if company in question:
            inferred["stock_code"] = company_to_code[company]
            inferred["matched_company"] = company
            break

    return inferred


# 合并用户显式传入的过滤条件和问题中自动识别的过滤条件。
def resolve_filters(question, metadata_items, stock_code=None, year=None, years=None):
    """Merge explicit filters with filters inferred from the question."""
    inferred = infer_filters_from_question(question, metadata_items)
    explicit_years = sorted(normalize_years(years) or [])
    resolved_year = year or (None if explicit_years else inferred["year"])
    resolved_years = explicit_years or inferred["years"]
    return {
        "stock_code": stock_code or inferred["stock_code"],
        "year": resolved_year,
        "years": resolved_years,
        "matched_company": inferred["matched_company"],
        "inferred_stock_code": inferred["stock_code"],
        "inferred_year": inferred["year"],
        "inferred_years": inferred["years"],
    }


# 根据过滤条件生成允许参与向量召回的 vector_id 集合。
def build_allowed_vector_ids(metadata_items, stock_code=None, year=None, years=None):
    """Build allowed vector_id set from optional filters."""
    if not stock_code and not year and not years:
        return None
    return {
        item["vector_id"]
        for item in metadata_items
        if item_matches_filter(item, stock_code=stock_code, year=year, years=years)
    }


# 用 FAISS 根据 query embedding 召回语义相似 chunks。
def vector_recall(index, metadata_items, query_vector, vector_top_k, stock_code=None, year=None, years=None):
    """
    Recall candidates from FAISS.

    Without filters, search exactly vector_top_k. With filters, search the whole
    small local index and then keep the first vector_top_k matched candidates.
    """
    allowed_ids = build_allowed_vector_ids(metadata_items, stock_code=stock_code, year=year, years=years)
    search_n = index.ntotal if allowed_ids is not None else min(index.ntotal, vector_top_k)
    scores, indices = index.search(query_vector, search_n)

    hits = []
    for score, vector_id in zip(scores[0], indices[0]):
        vector_id = int(vector_id)
        if vector_id < 0:
            continue
        if allowed_ids is not None and vector_id not in allowed_ids:
            continue
        item = metadata_items[vector_id]
        hits.append(
            {
                "chunk_id": item["chunk_id"],
                "vector_id": vector_id,
                "vector_rank": len(hits) + 1,
                "vector_score": float(score),
                "item": item,
            }
        )
        if len(hits) >= vector_top_k:
            break
    return hits


# 用 jieba 对中文文本分词，供 BM25 使用。
def tokenize_for_bm25(text):
    """Tokenize Chinese text for BM25."""
    return [token.strip() for token in jieba.lcut(text or "") if token.strip()]


# 用 BM25 根据关键词匹配召回候选 chunks。
def bm25_recall(metadata_items, question, top_n, stock_code=None, year=None, years=None):
    """Build BM25 online over filtered chunk.content and recall top_n."""
    candidates = [
        item
        for item in metadata_items
        if item_matches_filter(item, stock_code=stock_code, year=year, years=years)
    ]
    if not candidates:
        return []

    tokenized_docs = [tokenize_for_bm25(item.get("content", "")) for item in candidates]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = tokenize_for_bm25(question)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1]

    hits = []
    for idx in ranked_indices[:top_n]:
        score = float(scores[idx])
        if score <= 0:
            continue
        item = candidates[int(idx)]
        hits.append(
            {
                "chunk_id": item["chunk_id"],
                "vector_id": item["vector_id"],
                "bm25_rank": len(hits) + 1,
                "bm25_score": score,
                "item": item,
            }
        )
    return hits


# 用 RRF 融合向量召回和 BM25 召回排序。
def rrf_fusion(vector_hits, bm25_hits, rrf_k=DEFAULT_RRF_K):
    """Fuse vector and BM25 candidates with Reciprocal Rank Fusion."""
    fused = {}

    for hit in vector_hits:
        chunk_id = hit["chunk_id"]
        entry = fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "vector_id": hit["vector_id"],
                "item": hit["item"],
                "sources": [],
                "vector_rank": None,
                "vector_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            },
        )
        entry["sources"].append("vector")
        entry["vector_rank"] = hit["vector_rank"]
        entry["vector_score"] = hit["vector_score"]
        entry["rrf_score"] += 1.0 / (rrf_k + hit["vector_rank"])

    for hit in bm25_hits:
        chunk_id = hit["chunk_id"]
        entry = fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "vector_id": hit["vector_id"],
                "item": hit["item"],
                "sources": [],
                "vector_rank": None,
                "vector_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            },
        )
        entry["sources"].append("bm25")
        entry["bm25_rank"] = hit["bm25_rank"]
        entry["bm25_score"] = hit["bm25_score"]
        entry["rrf_score"] += 1.0 / (rrf_k + hit["bm25_rank"])

    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


# 把 chunk 内容和元数据拼成 rerank 模型输入文本。
def format_document_for_rerank(item):
    """Build rerank document text with metadata plus content."""
    meta = item.get("meta", {})
    page_start = meta.get("page_start", "")
    page_end = meta.get("page_end", "")
    page_text = f"{page_start}-{page_end}" if page_start != page_end else str(page_start)
    return (
        f"公司：{meta.get('company_name', '')}\n"
        f"股票代码：{meta.get('stock_code', '')}\n"
        f"年份：{meta.get('year', '')}\n"
        f"页码：{page_text}\n"
        f"章节：{meta.get('section', '')}\n"
        f"内容：\n{item.get('content', '')}"
    )


# 从 DashScope rerank 响应中解析索引和相关性分数。
def extract_rerank_results(response):
    """Extract rerank results from DashScope response variants."""
    if hasattr(response, "status_code") and response.status_code != 200:
        message = getattr(response, "message", "") or str(response)
        raise RuntimeError(f"DashScope rerank failed: {message}")
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    if output is None:
        raise RuntimeError(f"DashScope rerank response missing output: {response}")
    results = output.get("results") if isinstance(output, dict) else getattr(output, "results", None)
    if results is None:
        raise RuntimeError(f"DashScope rerank response missing results: {response}")
    parsed = []
    for result in results:
        if isinstance(result, dict):
            index = result.get("index")
            score = result.get("relevance_score", result.get("score"))
        else:
            index = getattr(result, "index", None)
            score = getattr(result, "relevance_score", getattr(result, "score", None))
        parsed.append({"index": int(index), "rerank_score": float(score)})
    return parsed


# 调用 rerank 模型对融合候选精排，并按阈值过滤。
def rerank_candidates(question, candidates, model, api_key, threshold, top_k):
    """Rerank RRF candidates and filter low relevance results."""
    if not candidates:
        return []

    documents = [format_document_for_rerank(candidate["item"]) for candidate in candidates]
    response = TextReRank.call(
        model=model,
        query=question,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        api_key=api_key,
    )
    rerank_results = extract_rerank_results(response)

    reranked = []
    for result in rerank_results:
        candidate = dict(candidates[result["index"]])
        candidate["rerank_rank"] = len(reranked) + 1
        candidate["rerank_score"] = result["rerank_score"]
        if candidate["rerank_score"] >= threshold:
            reranked.append(candidate)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    for index, item in enumerate(reranked, start=1):
        item["rerank_rank"] = index
    return reranked[:top_k]


# 把最终候选格式化成带引用编号的 LLM 上下文片段。
def format_context_item(number, candidate):
    """Format one final context with citation number."""
    item = candidate["item"]
    meta = item.get("meta", {})
    page_start = meta.get("page_start", "")
    page_end = meta.get("page_end", "")
    page_text = f"{page_start}-{page_end}" if page_start != page_end else str(page_start)
    return (
        f"【{number}】\n"
        f"公司：{meta.get('company_name', '')}\n"
        f"股票代码：{meta.get('stock_code', '')}\n"
        f"年份：{meta.get('year', '')}\n"
        f"页码：{page_text}\n"
        f"章节：{meta.get('section', '')}\n"
        f"内容：\n{item.get('content', '')}"
    )


# 构建最终 LLM 上下文和 references 引用列表。
def build_context(final_candidates, max_context_chars):
    """Build numbered LLM context and reference mapping."""
    contexts = []
    references = []
    used_chars = 0

    for number, candidate in enumerate(final_candidates, start=1):
        context = format_context_item(number, candidate)
        if contexts and used_chars + len(context) > max_context_chars:
            break
        contexts.append(context)
        used_chars += len(context)

        item = candidate["item"]
        meta = item.get("meta", {})
        references.append(
            {
                "ref_id": number,
                "chunk_id": item.get("chunk_id", ""),
                "filename": meta.get("filename", ""),
                "company_name": meta.get("company_name", ""),
                "stock_code": meta.get("stock_code", ""),
                "year": meta.get("year", ""),
                "section": meta.get("section", ""),
                "page_start": meta.get("page_start", ""),
                "page_end": meta.get("page_end", ""),
                "rerank_score": candidate.get("rerank_score"),
                "rrf_score": candidate.get("rrf_score"),
                "vector_score": candidate.get("vector_score"),
                "bm25_score": candidate.get("bm25_score"),
            }
        )
    return "\n\n".join(contexts), references


# 从 DashScope 文本生成响应中提取最终回答文本。
def extract_generation_text(response):
    """Extract text from DashScope Generation response."""
    if hasattr(response, "status_code") and response.status_code != 200:
        message = getattr(response, "message", "") or str(response)
        raise RuntimeError(f"DashScope generation failed: {message}")

    payload = response
    if hasattr(response, "to_dict"):
        payload = response.to_dict()

    output = get_field(payload, "output")
    if output is None:
        raise RuntimeError(f"DashScope generation response missing output: {response}")

    text = find_first_text(output)
    if text:
        return text
    raise RuntimeError(f"Unable to extract generation text: {response}")


# 兼容 dict 和对象两种形式读取字段。
def get_field(obj, field):
    """Read a field from either dict-like or attribute-like SDK objects."""
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


# 递归查找响应结构中的第一段非空文本。
def find_first_text(obj):
    """
    Recursively find the first non-empty generated text.

    DashScope responses vary by SDK/model/result_format. Some return
    output.text, some return choices[0].message.content, and newer chat
    formats may return content as a list of text parts.
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj.strip()

    if isinstance(obj, list):
        pieces = [find_first_text(item) for item in obj]
        pieces = [piece for piece in pieces if piece]
        return "\n".join(pieces)

    for field in ("text", "content"):
        value = get_field(obj, field)
        text = find_first_text(value)
        if text:
            return text

    choices = get_field(obj, "choices")
    if choices:
        text = find_first_text(choices)
        if text:
            return text

    message = get_field(obj, "message")
    if message:
        text = find_first_text(message)
        if text:
            return text

    if isinstance(obj, dict):
        for key in ("output", "message", "choice"):
            text = find_first_text(obj.get(key))
            if text:
                return text

    return ""


# 调用 qwen-plus，根据检索上下文生成最终答案。
def call_llm(question, context, model, api_key):
    """Call qwen-plus with grounded context."""
    user_prompt = f"问题：\n{question}\n\n上下文：\n{context}\n\n请基于以上上下文回答。用一句完整的话回答，包含问题主体和指标，保持简洁。"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = Generation.call(
        model=model,
        messages=messages,
        api_key=api_key,
        result_format="message",
        temperature=0.2,
    )
    try:
        return extract_generation_text(response)
    except RuntimeError:
        fallback_response = Generation.call(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=0.2,
        )
        return extract_generation_text(fallback_response)


# 压缩候选信息，保留分数和预览，便于 debug 输出。
def compact_candidate_for_debug(candidate):
    """Remove full item content from debug candidate while preserving scores."""
    item = candidate.get("item", {})
    meta = item.get("meta", {})
    return {
        "chunk_id": candidate.get("chunk_id"),
        "vector_id": candidate.get("vector_id"),
        "sources": candidate.get("sources"),
        "vector_rank": candidate.get("vector_rank"),
        "vector_score": candidate.get("vector_score"),
        "bm25_rank": candidate.get("bm25_rank"),
        "bm25_score": candidate.get("bm25_score"),
        "rrf_score": candidate.get("rrf_score"),
        "rerank_rank": candidate.get("rerank_rank"),
        "rerank_score": candidate.get("rerank_score"),
        "meta": meta,
        "content_preview": item.get("content", "")[:200],
    }


# 串起完整 RAG 问答流程，是命令行问答主入口。
def answer_question(args):
    """Run the whole RAG pipeline."""
    api_key = get_api_key()
    index, metadata_items = load_index_and_metadata(args.index_path, args.metadata_path)
    resolved_filters = resolve_filters(
        question=args.question,
        metadata_items=metadata_items,
        stock_code=args.stock_code,
        year=args.year,
        years=getattr(args, "years", None),
    )

    query_vector = embed_query(args.question, args.embedding_model, api_key)
    vector_hits = vector_recall(
        index=index,
        metadata_items=metadata_items,
        query_vector=query_vector,
        vector_top_k=args.vector_top_k,
        stock_code=resolved_filters["stock_code"],
        year=resolved_filters["year"],
        years=resolved_filters["years"],
    )
    bm25_hits = bm25_recall(
        metadata_items=metadata_items,
        question=args.question,
        top_n=args.bm25_top_k,
        stock_code=resolved_filters["stock_code"],
        year=resolved_filters["year"],
        years=resolved_filters["years"],
    )
    rrf_hits = rrf_fusion(vector_hits, bm25_hits, rrf_k=args.rrf_k)
    rerank_input = rrf_hits[: args.rerank_candidates]
    final_candidates = rerank_candidates(
        question=args.question,
        candidates=rerank_input,
        model=args.rerank_model,
        api_key=api_key,
        threshold=args.rerank_threshold,
        top_k=args.top_k,
    )

    context, references = build_context(final_candidates, args.max_context_chars)
    if not context:
        answer = "根据已提供资料无法判断。"
    else:
        answer = call_llm(args.question, context, args.llm_model, api_key)

    result = {
        "question": args.question,
        "answer": answer,
        "references": references,
        "debug": {
            "filters": {
                "explicit_stock_code": args.stock_code,
                "explicit_year": args.year,
                "inferred_stock_code": resolved_filters["inferred_stock_code"],
                "inferred_year": resolved_filters["inferred_year"],
                "inferred_years": resolved_filters["inferred_years"],
                "matched_company": resolved_filters["matched_company"],
                "resolved_stock_code": resolved_filters["stock_code"],
                "resolved_year": resolved_filters["year"],
                "resolved_years": resolved_filters["years"],
            },
            "vector_hits": [compact_candidate_for_debug(hit) for hit in vector_hits],
            "bm25_hits": [compact_candidate_for_debug(hit) for hit in bm25_hits],
            "rrf_hits": [compact_candidate_for_debug(hit) for hit in rrf_hits],
            "rerank_hits": [compact_candidate_for_debug(hit) for hit in final_candidates],
        },
    }
    return result


# 把答案和引用来源打印到命令行。
def print_result(result):
    """Print answer and references to console."""
    filters = result.get("debug", {}).get("filters", {})
    if filters:
        print(
            f"过滤条件：stock_code={filters.get('resolved_stock_code') or '无'}, "
            f"year={filters.get('resolved_year') or '无'}"
        )
    print("\n答案：")
    print(result["answer"])
    if result["references"]:
        print("\n引用：")
        for ref in result["references"]:
            page_start = ref.get("page_start", "")
            page_end = ref.get("page_end", "")
            page_text = f"{page_start}-{page_end}" if page_start != page_end else str(page_start)
            print(
                f"【{ref['ref_id']}】{ref.get('company_name', '')} {ref.get('year', '')} "
                f"第{page_text}页 {ref.get('section', '')} "
                f"(chunk_id={ref.get('chunk_id', '')}, rerank={ref.get('rerank_score'):.4f})"
            )


# 命令行入口：解析参数并执行 answer_question。
def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run financial annual-report RAG QA.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="Path to faiss.index.")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, help="Path to metadata.json.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="DashScope embedding model.")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL, help="DashScope rerank model.")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="DashScope LLM model.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Final context count.")
    parser.add_argument("--vector-top-k", type=int, default=DEFAULT_VECTOR_TOP_K, help="Vector recall count.")
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K, help="BM25 recall count.")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF k constant.")
    parser.add_argument("--rerank-candidates", type=int, default=DEFAULT_RERANK_CANDIDATES, help="Candidates sent to reranker.")
    parser.add_argument("--rerank-threshold", type=float, default=DEFAULT_RERANK_THRESHOLD, help="Min rerank score.")
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS, help="Max context chars.")
    parser.add_argument("--stock-code", default=None, help="Optional stock code filter.")
    parser.add_argument("--year", default=None, help="Optional year filter.")
    parser.add_argument("--debug-output", default=None, help="Optional path to save full result JSON.")
    args = parser.parse_args()

    result = answer_question(args)
    print_result(result)
    if args.debug_output:
        write_json(args.debug_output, result)
        print(f"\nDebug saved to: {args.debug_output}")


if __name__ == "__main__":
    main()
