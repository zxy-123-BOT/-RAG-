"""
年报解析结果切片脚本。

本文件读取 data/parsed_json 中的 parsed JSON，把正文和表格整理成适合 RAG 检索的 chunks。
输出结果写入 data/chunks，其中 all_chunks.json 会作为 build_index.py 的输入。

运行示例:
    python src/chunk_reports.py
    python src/chunk_reports.py --max-text-chars 1000
"""

import argparse
import json
import re
from pathlib import Path


DEFAULT_MAX_TEXT_CHARS = 1000
TABLE_CONTEXT_MAX_CHARS = 120


# 规范 chunk 文本空白，同时保留有用换行。
def normalize_text(text):
    """Normalize text spacing while keeping useful line breaks."""
    text = re.sub(r"[ \t\r\f\v]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 清理文件名非法字符，保证输出路径可写。
def safe_filename_part(text):
    """Make metadata safe to use as part of a Windows filename."""
    text = re.sub(r'[\\/:*?"<>|]+', "_", text or "")
    return text.strip("_ ") or "unknown"


# 从 block metadata 中读取 block 类型。
def get_block_type(block):
    """Return parsed block type: title, text, or table."""
    return block.get("meta", {}).get("block_type", "")


# 从 block metadata 中读取章节路径。
def get_section(block):
    """Return section string from block metadata."""
    return block.get("meta", {}).get("section", "")


# 从 block metadata 中读取页码并转为整数。
def get_page(block):
    """Return 1-based page number from block metadata."""
    return int(block.get("meta", {}).get("page") or 0)


# 判断短文本是否是表格前的单位或币种说明。
def is_unit_note(text):
    """Return True if text is a short unit/currency note before a table."""
    if len(text) > TABLE_CONTEXT_MAX_CHARS:
        return False
    patterns = [
        r"单位[:：]",
        r"币种[:：]",
        r"金额单位",
        r"货币单位",
        r"人民币",
        r"元\s*$",
        r"万元\s*$",
        r"千元\s*$",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


# 判断短文本是否像表格标题。
def is_short_table_title(text):
    """Return True if text looks like a short table title."""
    if not text or len(text) > 60:
        return False
    if "适用" in text:
        return False
    if text.endswith(("。", "；", ";", "，", ",")):
        return False
    if "\n" in text and len(text) > 35:
        return False
    return True


# 判断表格前的标题或单位说明是否应合并进表格 chunk。
def can_absorb_before_table(candidate, table_block):
    """
    Decide whether a block immediately before a table should be added to the
    table chunk.

    This is deliberately conservative: only short table titles and unit notes
    are absorbed, so ordinary text does not get mixed into table chunks.
    """
    block_type = get_block_type(candidate)
    text = normalize_text(candidate.get("content", ""))
    if not text:
        return False
    if get_section(candidate) != get_section(table_block):
        return False
    if abs(get_page(table_block) - get_page(candidate)) > 1:
        return False
    if block_type == "title":
        return is_short_table_title(text)
    if block_type == "text":
        return is_unit_note(text)
    return False


# 按指定正则边界切分长文本，并尽量保留边界符号。
def split_by_pattern(text, max_chars, pattern):
    """Split text by regex delimiters while keeping the delimiter text."""
    pieces = re.split(pattern, text)
    chunks = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(current) + len(piece) <= max_chars:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


# 按段落、句号、换行等语义边界切分过长正文。
def split_long_text(text, max_chars):
    """
    Split long text by semantic boundaries.

    The splitter tries paragraph breaks first, then Chinese sentence endings,
    then line breaks. Only when no semantic boundary is available does it fall
    back to fixed-length slicing.
    """
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    candidate_chunks = [text]
    for pattern in [r"(?<=\n\n)", r"(?<=[。；;])", r"(?<=\n)"]:
        next_chunks = []
        changed = False
        for chunk in candidate_chunks:
            if len(chunk) <= max_chars:
                next_chunks.append(chunk)
                continue
            pieces = split_by_pattern(chunk, max_chars, pattern)
            next_chunks.extend(pieces)
            changed = changed or len(pieces) > 1
        candidate_chunks = next_chunks
        if changed and all(len(chunk) <= max_chars for chunk in candidate_chunks):
            return [chunk for chunk in candidate_chunks if chunk]

    final_chunks = []
    for chunk in candidate_chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue
        for start in range(0, len(chunk), max_chars):
            final_chunks.append(chunk[start : start + max_chars].strip())
    return [chunk for chunk in final_chunks if chunk]


# 从源 blocks 继承公司、年份、章节、页码范围等 chunk 元数据。
def base_meta_from_blocks(blocks, chunk_type):
    """Build chunk metadata by inheriting common metadata from source blocks."""
    first_meta = blocks[0]["meta"]
    pages = [get_page(block) for block in blocks if get_page(block)]
    return {
        "chunk_type": chunk_type,
        "filename": first_meta.get("filename", ""),
        "company_name": first_meta.get("company_name", ""),
        "stock_code": first_meta.get("stock_code", ""),
        "year": first_meta.get("year", ""),
        "section": first_meta.get("section", ""),
        "page_start": min(pages) if pages else 0,
        "page_end": max(pages) if pages else 0,
    }


# 生成统一 chunk 结构。
def make_chunk(chunk_id, content, meta):
    """Create one final chunk object."""
    return {
        "chunk_id": chunk_id,
        "content": normalize_text(content),
        "meta": meta,
    }


# 基于股票代码、年份和序号生成稳定 chunk_id。
def chunk_id_for(doc_meta, index):
    """Create a stable chunk id for one document."""
    stock_code = doc_meta.get("stock_code") or safe_filename_part(doc_meta.get("company_name"))
    year = doc_meta.get("year") or "unknown"
    return f"{stock_code}_{year}_{index:06d}"


# 构建合并文本与源 block 的字符范围映射。
def build_text_buffer_ranges(text_buffer):
    """Build combined text and char ranges for each source text block."""
    pieces = []
    ranges = []
    cursor = 0
    for item in text_buffer:
        text = normalize_text(item["block"].get("content", ""))
        if not text:
            continue
        if pieces:
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        ranges.append(
            {
                "start": start,
                "end": cursor,
                "block": item["block"],
            }
        )
    return "\n\n".join(pieces), ranges


# 找到切分后某段文本覆盖到的源 blocks，用于保留准确页码。
def blocks_for_text_part(part, combined_text, ranges, search_start):
    """
    Find source blocks covered by one split text part.

    The splitter preserves original text order, so a moving search cursor is
    enough to map each split part back to the source block ranges and produce
    accurate page_start/page_end for that chunk.
    """
    start = combined_text.find(part, search_start)
    if start < 0:
        start = search_start
    end = start + len(part)
    blocks = [item["block"] for item in ranges if item["start"] < end and item["end"] > start]
    return blocks, end


# 把暂存的正文 blocks 输出成一个或多个 text chunks。
def flush_text_buffer(text_buffer, chunks, doc_meta, max_text_chars):
    """Convert buffered text blocks into one or more text chunks."""
    if not text_buffer:
        return 0
    content, ranges = build_text_buffer_ranges(text_buffer)
    parts = split_long_text(content, max_text_chars)
    created = 0
    search_start = 0
    for part in parts:
        chunk_index = len(chunks) + 1
        part_blocks, search_start = blocks_for_text_part(part, content, ranges, search_start)
        if not part_blocks:
            part_blocks = [item["block"] for item in text_buffer]
        meta = base_meta_from_blocks(part_blocks, "text")
        chunks.append(make_chunk(chunk_id_for(doc_meta, chunk_index), part, meta))
        created += 1
    text_buffer.clear()
    return created


# 从正文暂存区移除已经并入表格 chunk 的说明文本。
def remove_absorbed_from_buffer(text_buffer, absorbed_indices):
    """Remove absorbed text blocks from the pending text buffer."""
    if not absorbed_indices:
        return []
    absorbed = []
    remaining = []
    for item in text_buffer:
        if item["index"] in absorbed_indices:
            absorbed.append(item["block"])
        else:
            remaining.append(item)
    text_buffer[:] = remaining
    return absorbed


# 查找表格前最多两个可吸收的标题或单位说明 blocks。
def find_table_context(blocks, table_index, text_buffer):
    """
    Find at most two blocks immediately before a table as table context.

    Title blocks are never standalone chunks, so they can be safely absorbed.
    Text blocks are absorbed only if they are still pending in text_buffer;
    this prevents duplicate content in text chunks and table chunks.
    """
    table_block = blocks[table_index]
    pending_text_indices = {item["index"] for item in text_buffer}
    context_blocks = []
    absorbed_text_indices = set()

    start = max(0, table_index - 2)
    for index in range(start, table_index):
        candidate = blocks[index]
        block_type = get_block_type(candidate)
        if not can_absorb_before_table(candidate, table_block):
            continue
        if block_type == "text" and index not in pending_text_indices:
            continue
        context_blocks.append(candidate)
        if block_type == "text":
            absorbed_text_indices.add(index)

    remove_absorbed_from_buffer(text_buffer, absorbed_text_indices)
    return context_blocks


# 估算 Markdown 表格的列数，用于判断跨页表格是否连续。
def markdown_column_count(table_text):
    """Estimate markdown table column count from the first table row."""
    for line in normalize_text(table_text).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        return len(cells)
    return 0


# 判断相邻两页表格是否属于同一个跨页表格。
def should_merge_table_continuation(previous_table, next_table):
    """
    Decide whether two adjacent table blocks are one cross-page table.

    We only merge when the next table is on the next page, belongs to the same
    section, and has the same markdown column count. Same-page tables are kept
    separate because they are often different tables under the same section.
    """
    if get_section(previous_table) != get_section(next_table):
        return False
    if get_page(next_table) != get_page(previous_table) + 1:
        return False
    prev_cols = markdown_column_count(previous_table.get("content", ""))
    next_cols = markdown_column_count(next_table.get("content", ""))
    return prev_cols > 0 and prev_cols == next_cols


# 从当前表格开始收集连续跨页表格 blocks。
def collect_table_run(blocks, table_index):
    """Collect consecutive cross-page table blocks starting at table_index."""
    table_blocks = [blocks[table_index]]
    next_index = table_index + 1
    while next_index < len(blocks):
        candidate = blocks[next_index]
        if get_block_type(candidate) != "table":
            break
        if not should_merge_table_continuation(table_blocks[-1], candidate):
            break
        table_blocks.append(candidate)
        next_index += 1
    return table_blocks, next_index


# 把表格和可选上下文说明合并成 table chunk。
def build_table_chunk(table_blocks, context_blocks, chunks, doc_meta):
    """Create one table chunk with optional short title/unit context."""
    chunk_index = len(chunks) + 1
    source_blocks = [*context_blocks, *table_blocks]
    context_text = "\n\n".join(normalize_text(block.get("content", "")) for block in context_blocks)
    table_text = "\n\n".join(normalize_text(block.get("content", "")) for block in table_blocks)
    content = f"{context_text}\n\n{table_text}" if context_text else table_text
    meta = base_meta_from_blocks(source_blocks, "table")
    chunks.append(make_chunk(chunk_id_for(doc_meta, chunk_index), content, meta))


# 对单份 parsed JSON 生成 chunks。
def build_chunks_for_doc(parsed, max_text_chars):
    """Build chunks for one parsed annual report."""
    doc_meta = parsed.get("doc_meta", {})
    blocks = parsed.get("blocks", [])
    chunks = []
    text_buffer = []

    index = 0
    while index < len(blocks):
        block = blocks[index]
        block_type = get_block_type(block)
        if block_type == "title":
            flush_text_buffer(text_buffer, chunks, doc_meta, max_text_chars)
            index += 1
            continue

        if block_type == "text":
            if text_buffer and get_section(text_buffer[-1]["block"]) != get_section(block):
                flush_text_buffer(text_buffer, chunks, doc_meta, max_text_chars)
            text_buffer.append({"index": index, "block": block})
            index += 1
            continue

        if block_type == "table":
            context_blocks = find_table_context(blocks, index, text_buffer)
            table_blocks, next_index = collect_table_run(blocks, index)
            flush_text_buffer(text_buffer, chunks, doc_meta, max_text_chars)
            build_table_chunk(table_blocks, context_blocks, chunks, doc_meta)
            index = next_index
            continue

        index += 1

    flush_text_buffer(text_buffer, chunks, doc_meta, max_text_chars)
    return {
        "doc_meta": doc_meta,
        "chunks": chunks,
    }


# 为单份年报 chunks 生成输出文件路径。
def output_path_for(parsed_path, parsed, output_dir):
    """Build per-document chunk JSON path."""
    meta = parsed.get("doc_meta", {})
    stock_code = meta.get("stock_code") or "unknown"
    year = meta.get("year") or "unknown"
    company = safe_filename_part(meta.get("company_name") or parsed_path.stem)
    return output_dir / f"{stock_code}_{year}_{company}_chunks.json"


# 批量处理 parsed_json 目录，并汇总生成 all_chunks.json。
def chunk_all(input_dir, output_dir, max_text_chars=DEFAULT_MAX_TEXT_CHARS, limit=None):
    """Chunk all parsed JSON files and write per-file plus all_chunks outputs."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_paths = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if limit:
        parsed_paths = parsed_paths[:limit]

    all_chunks = []
    output_paths = []
    for index, parsed_path in enumerate(parsed_paths, start=1):
        print(f"[{index}/{len(parsed_paths)}] Chunking {parsed_path.name} ...")
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        chunked = build_chunks_for_doc(parsed, max_text_chars)
        out_path = output_path_for(parsed_path, parsed, output_dir)
        out_path.write_text(json.dumps(chunked, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {out_path} ({len(chunked['chunks'])} chunks)")
        output_paths.append(out_path)
        all_chunks.extend(chunked["chunks"])

    all_path = output_dir / "all_chunks.json"
    all_path.write_text(json.dumps({"chunks": all_chunks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {all_path} ({len(all_chunks)} chunks total)")
    return output_paths, all_path


# 命令行入口：解析参数并启动批量切片。
def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build RAG chunks from parsed annual-report JSON files.")
    parser.add_argument("--input-dir", default="data/parsed_json", help="Directory containing parsed JSON files.")
    parser.add_argument("--output-dir", default="data/chunks", help="Directory for chunk JSON files.")
    parser.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS, help="Max chars per text chunk.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N parsed JSON files.")
    args = parser.parse_args()

    chunk_all(args.input_dir, args.output_dir, args.max_text_chars, args.limit)


if __name__ == "__main__":
    main()
