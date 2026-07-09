"""
向量索引构建脚本。

这个文件读取 data/chunks/all_chunks.json，调用 DashScope embedding 模型生成向量，
然后构建 FAISS IndexFlatIP 索引，并写出 metadata、配置、缓存和失败记录。

整体流程:
1. 读取 all_chunks.json，并过滤空内容 chunk。
2. 从 embedding_cache.jsonl 读取已有 embedding，支持断点续建。
3. 对未缓存 chunk 批量调用 text-embedding-v3。
4. 每个成功 batch 立即追加写入缓存，避免中断后重来。
5. 把 embedding 组装成 numpy 矩阵，并做 L2 归一化。
6. 构建 FAISS IndexFlatIP，用内积实现 cosine 相似度搜索。
7. 写出 faiss.index、metadata.json、index_config.json、failed_chunks.json。

运行示例:
    python src/build_index.py
    python src/build_index.py --limit 20
    python src/build_index.py --rebuild
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise RuntimeError("缺少 faiss, 请先安装 faiss-cpu。") from exc


DEFAULT_MODEL = "text-embedding-v3"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_SLEEP_SECONDS = 0.5


# 返回进度条对象；如果安装了 tqdm 就显示可视化进度，否则直接返回原 iterable。
def get_progress_bar(iterable, total=None, desc="progress", unit="it"):
    """
    Return a tqdm progress bar when tqdm is installed.

    tqdm is optional. If it is not installed, the script still runs and falls
    back to the original iterable without a visual progress bar.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


# 延迟导入 DashScope 的 TextEmbedding 客户端，让脚本在未真正构建索引前也能被安全 import。
def load_dashscope_embedding_client():
    """
    Import DashScope SDK lazily.

    This keeps the script importable even when dashscope is not installed, and
    gives a clear message only when the embedding build actually runs.
    """
    try:
        from dashscope import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "缺少 dashscope SDK, 请先运行: python -m pip install dashscope"
        ) from exc
    return TextEmbedding


# 读取 UTF-8 JSON 文件，并把内容解析成 Python 对象。
def read_json(path):
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 把 Python 对象写成格式化后的 UTF-8 JSON 文件，便于人工查看。
def write_json(path, data):
    """Write a UTF-8 JSON file with readable indentation."""
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 定义每个 chunk 送入 embedding 模型的文本；当前只使用 chunk.content。
def chunk_embedding_text(chunk):
    """Return the exact text sent to the embedding model."""
    return (chunk.get("content") or "").strip()


# 读取 all_chunks.json，并过滤掉 content 为空的 chunk；limit 用于小规模测试。
def load_chunks(chunks_path, limit=None):
    """Load chunks and drop empty-content items."""
    data = read_json(chunks_path)
    chunks = data.get("chunks", [])
    if limit:
        chunks = chunks[:limit]
    return [chunk for chunk in chunks if chunk_embedding_text(chunk)]


# 生成一条与 FAISS vector_id 对齐的 metadata，后续检索命中 vector_id 后靠它还原 chunk 信息。
def make_metadata_item(vector_id, chunk):
    """Build one metadata item aligned with a FAISS vector id."""
    return {
        "vector_id": vector_id,
        "chunk_id": chunk.get("chunk_id", ""),
        "content": chunk.get("content", ""),
        "meta": chunk.get("meta", {}),
    }


# 读取 embedding_cache.jsonl 缓存，按 chunk_id 建立索引，支持断点续建。
def read_embedding_cache(cache_path):
    """Read JSONL embedding cache keyed by chunk_id."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}

    cache = {}
    with cache_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunk_id = record.get("chunk_id")
            if chunk_id:
                cache[chunk_id] = record
    return cache


# 把一批成功生成的 embedding 立即追加写入 JSONL 缓存，降低中途中断的损失。
def append_embedding_cache(cache_path, records):
    """Append successful embedding records to JSONL cache immediately."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 从 DashScope embedding 响应中取出向量，兼容 dict 和 SDK 对象两种返回格式。
def extract_embeddings_from_response(response):
    """
    Extract embeddings from DashScope response variants.

    DashScope SDK versions may expose response fields as dict-like objects or
    attributes. This helper keeps the build script tolerant of both formats.
    """
    if hasattr(response, "status_code") and response.status_code != 200:
        message = getattr(response, "message", "") or str(response)
        raise RuntimeError(f"DashScope embedding failed: {message}")

    output = None
    if isinstance(response, dict):
        output = response.get("output")
    else:
        output = getattr(response, "output", None)

    if output is None:
        raise RuntimeError(f"DashScope response missing output: {response}")

    embeddings = None
    if isinstance(output, dict):
        embeddings = output.get("embeddings")
    else:
        embeddings = getattr(output, "embeddings", None)

    if not embeddings:
        raise RuntimeError(f"DashScope response missing embeddings: {response}")

    result = []
    for item in embeddings:
        if isinstance(item, dict):
            result.append(item["embedding"])
        else:
            result.append(item.embedding)
    return result


# 调用百炼/DashScope embedding API，为一批文本生成 embedding。
def call_embedding_api(texts, model, api_key):
    """Call Bailian/DashScope text embedding API for a batch of texts."""
    TextEmbedding = load_dashscope_embedding_client()
    response = TextEmbedding.call(
        model=model,
        input=texts,
        api_key=api_key,
    )
    embeddings = extract_embeddings_from_response(response)
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
    return embeddings


# 对单个 batch 调用 embedding API；失败时按次数重试，并做简单线性退避。
def embed_batch_with_retries(texts, model, api_key, max_retries, sleep_seconds):
    """Embed one batch with retry and simple backoff."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_embedding_api(texts, model, api_key)
        except Exception as exc:
            last_error = exc
            wait = sleep_seconds * attempt
            print(f"  embedding retry {attempt}/{max_retries}: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"Embedding failed after {max_retries} retries: {last_error}")


# 构建或续建 embedding 缓存：跳过已缓存 chunk，只请求缺失部分，并记录失败项。
def build_embedding_cache(chunks, cache_path, failed_path, model, batch_size, max_retries, sleep_seconds, rebuild):
    """
    Build or resume embedding cache.

    Every successful batch is written to disk before continuing, so interrupting
    the process does not waste completed API calls.
    """
    cache_path = Path(cache_path)
    failed_path = Path(failed_path)
    if rebuild and cache_path.exists():
        cache_path.unlink()
    if rebuild and failed_path.exists():
        failed_path.unlink()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY。")

    cache = read_embedding_cache(cache_path)
    failed = []
    pending = [chunk for chunk in chunks if chunk.get("chunk_id") not in cache]
    print(f"cached embeddings: {len(cache)}")
    print(f"pending embeddings: {len(pending)}")

    batch_starts = list(range(0, len(pending), batch_size))
    progress = get_progress_bar(batch_starts, total=len(batch_starts), desc="Embedding", unit="batch")
    completed_chunks = 0
    for start in progress:
        batch = pending[start : start + batch_size]
        texts = [chunk_embedding_text(chunk) for chunk in batch]
        batch_no = start // batch_size + 1
        batch_count = len(batch_starts)
        if not hasattr(progress, "set_postfix"):
            print(f"embedding batch {batch_no}/{batch_count} ({len(batch)} chunks)")
        try:
            embeddings = embed_batch_with_retries(texts, model, api_key, max_retries, sleep_seconds)
        except Exception as exc:
            for chunk in batch:
                failed.append(
                    {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "error": str(exc),
                    }
                )
            completed_chunks += len(batch)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(done=completed_chunks, failed=len(failed))
            continue

        records = []
        for chunk, embedding in zip(batch, embeddings):
            record = {
                "chunk_id": chunk.get("chunk_id", ""),
                "embedding": embedding,
            }
            records.append(record)
            cache[record["chunk_id"]] = record
        append_embedding_cache(cache_path, records)
        completed_chunks += len(batch)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(done=completed_chunks, failed=len(failed))
        time.sleep(sleep_seconds)

    write_json(failed_path, {"failed": failed})
    return read_embedding_cache(cache_path), failed


# 根据 chunks 和 embedding 缓存生成向量矩阵与 metadata 列表，两者顺序必须严格一致。
def build_vectors_and_metadata(chunks, cache):
    """Build numpy matrix and metadata list in the same order."""
    vectors = []
    metadata_items = []
    missing = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        record = cache.get(chunk_id)
        if not record:
            missing.append(chunk_id)
            continue
        vector_id = len(vectors)
        vectors.append(record["embedding"])
        metadata_items.append(make_metadata_item(vector_id, chunk))

    if not vectors:
        raise RuntimeError("没有可用于构建索引的 embedding。")

    matrix = np.asarray(vectors, dtype="float32")
    return matrix, metadata_items, missing


# 对向量做 L2 归一化，并构建 FAISS IndexFlatIP 索引，用内积近似 cosine 相似度。
def build_faiss_index(vectors):
    """Build IndexFlatIP after L2 normalization."""
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")
    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


# 串起完整索引构建流程：加载 chunk、生成/读取 embedding、建 FAISS、写 metadata 和配置。
def build_index(args):
    """Build embeddings, FAISS index, metadata, and config files."""
    chunks_path = Path(args.chunks_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss_index_path = output_dir / "faiss.index"
    metadata_path = output_dir / "metadata.json"
    config_path = output_dir / "index_config.json"
    cache_path = output_dir / "embedding_cache.jsonl"
    failed_path = output_dir / "failed_chunks.json"

    chunks = load_chunks(chunks_path, args.limit)
    print(f"loaded chunks: {len(chunks)}")

    cache, failed = build_embedding_cache(
        chunks=chunks,
        cache_path=cache_path,
        failed_path=failed_path,
        model=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep,
        rebuild=args.rebuild,
    )

    vectors, metadata_items, missing = build_vectors_and_metadata(chunks, cache)
    index = build_faiss_index(vectors)

    faiss.write_index(index, str(faiss_index_path))
    write_json(metadata_path, {"items": metadata_items})

    config = {
        "embedding_provider": "dashscope_bailian",
        "embedding_model": args.model,
        "embedding_text": "chunk.content",
        "index_type": "faiss.IndexFlatIP",
        "normalize_l2": True,
        "dimension": int(vectors.shape[1]),
        "chunk_count_input": len(chunks),
        "chunk_count_indexed": len(metadata_items),
        "failed_count": len(failed),
        "missing_count": len(missing),
        "source_chunks_path": str(chunks_path),
        "faiss_index_path": str(faiss_index_path),
        "metadata_path": str(metadata_path),
        "cache_path": str(cache_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(config_path, config)

    if missing:
        print(f"missing embeddings skipped: {len(missing)}")
    print(f"index vectors: {index.ntotal}")
    print(f"dimension: {vectors.shape[1]}")
    print(f"wrote: {faiss_index_path}")
    print(f"wrote: {metadata_path}")
    print(f"wrote: {config_path}")


# 命令行入口：解析参数，并调用 build_index 执行索引构建。
def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build FAISS IndexFlatIP from RAG chunks using Bailian embeddings.")
    parser.add_argument("--chunks-path", default="data/chunks/all_chunks.json", help="Path to all_chunks.json.")
    parser.add_argument("--output-dir", default="data/index", help="Directory for FAISS index and metadata.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DashScope embedding model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries per batch.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Sleep seconds between batches.")
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N chunks for testing.")
    parser.add_argument("--rebuild", action="store_true", help="Delete cache and rebuild embeddings.")
    args = parser.parse_args()

    build_index(args)


if __name__ == "__main__":
    main()
