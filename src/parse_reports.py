"""
年报 PDF 解析脚本。

本文件负责把 data/raw_pdf 中的年报 PDF 解析成结构化 JSON，输出到 data/parsed_json。
解析时会尽量保留公司、股票代码、年份、章节、页码、正文块、表格块等信息，供后续 chunk_reports.py 切片使用。

运行示例:
    python src/parse_reports.py
    python src/parse_reports.py --limit 1
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

import fitz
import pdfplumber


# 公司名到股票代码的兜底映射。优先从 PDF 首页/前几页提取证券代码,
# 只有提取不到时才使用这里的映射。
COMPANY_CODE_MAP = {
    "中芯国际": "688981",
    "奥来德": "688378",
    "智飞生物": "300122",
    "良品铺子": "603719",
    "长电科技": "600584",
}

# 标题层级规则:
# level 1: 第一节 ...
# level 2: 一、...
# level 3: （一）...
# level 4: 1....
# level 5: （1）...
SECTION_PATTERNS = [
    (1, re.compile(r"^第[一二三四五六七八九十百\d]+节\s*\S+")),
    (2, re.compile(r"^[一二三四五六七八九十]+[、.．]\s*\S+")),
    (3, re.compile(r"^[（(][一二三四五六七八九十]+[）)]\s*\S+")),
    (4, re.compile(r"^\d+[、.．]\s*\S+")),
    (5, re.compile(r"^[（(]\d+[）)]\s*\S+")),
]

PAGE_NUMBER_RE = re.compile(r"^\s*(?:第\s*)?\d+\s*(?:页)?(?:\s*/\s*\d+)?\s*$")
YEAR_RE = re.compile(r"(20\d{2})")
STOCK_CODE_RE = re.compile(r"(?:股票|证券|A股)\s*(?:代码|简称)?\s*[:：]?\s*([03668]\d{5})")


# 规范文本空白，去掉多余空格和连续空行，方便后续解析。
def normalize_text(text):
    """Normalize spaces and excessive blank lines."""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 去掉所有空白字符，用于稳定比较页眉、页脚、页码等噪声文本。
def compact_text(text):
    """Remove all whitespace for stable header/footer comparison."""
    return re.sub(r"\s+", "", text or "")


# 清理 Windows 文件名非法字符，保证公司名等字段能安全写入文件名。
def safe_filename_part(text):
    """Make metadata safe to use as part of a Windows filename."""
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    return text.strip("_ ") or "unknown"


# 从常见年报文件名格式中推断公司简称。
def extract_company_from_filename(path):
    """Infer company name from common annual-report filename patterns."""
    stem = path.stem
    if "：" in stem:
        return stem.split("：", 1)[0].strip()
    if ":" in stem:
        return stem.split(":", 1)[0].strip()
    match = re.match(r"(.+?)(?:20\d{2}|年报|年度报告)", stem)
    if match:
        return match.group(1).strip("_ -")
    return stem


# 从文件名和 PDF 前几页提取文档级元数据，例如公司、股票代码、年份。
def extract_doc_meta(pdf_path, doc):
    """
    Extract document-level metadata.

    The first page and the first few pages often contain company name,
    report year, and stock code. These fields are stored in doc_meta only;
    the cover page itself will not become a retrievable block.
    """
    filename = pdf_path.name
    first_pages_text = []
    for page_index in range(min(3, len(doc))):
        first_pages_text.append(doc[page_index].get_text("text"))
    probe_text = "\n".join(first_pages_text)

    company_name = extract_company_from_filename(pdf_path)
    year_match = YEAR_RE.search(filename) or YEAR_RE.search(probe_text)
    code_match = STOCK_CODE_RE.search(probe_text)

    stock_code = code_match.group(1) if code_match else COMPANY_CODE_MAP.get(company_name, "")
    year = year_match.group(1) if year_match else ""

    return {
        "filename": filename,
        "company_name": company_name,
        "stock_code": stock_code,
        "year": year,
        "report_type": "annual_report",
        "source_path": str(pdf_path),
    }


# 判断某一页是否像目录页，避免目录内容进入检索语料。
def is_toc_like_page(text):
    """Return True when a page looks like a table of contents."""
    clean = normalize_text(text)
    if not clean:
        return False
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    section_lines = 0
    dotted_lines = 0
    for line in lines:
        if re.search(r"第[一二三四五六七八九十百\d]+节", line):
            section_lines += 1
        if re.search(r"\.{2,}|…{2,}|·{2,}|\s{3,}\d+\s*$", line):
            dotted_lines += 1
    if "目录" in clean[:200] and (section_lines >= 2 or dotted_lines >= 3):
        return True
    return section_lines >= 4 and dotted_lines >= 2


# 扫描 PDF 前几页，找出需要跳过的目录页页码。
def detect_toc_pages(doc, max_scan_pages=12):
    """
    Detect TOC pages near the beginning of the PDF.

    TOC pages are useful for understanding structure, but they are skipped
    from blocks because they duplicate section titles and hurt retrieval.
    """
    toc_pages = set()
    in_toc = False
    for page_index in range(1, min(max_scan_pages, len(doc))):
        text = doc[page_index].get_text("text")
        if is_toc_like_page(text):
            toc_pages.add(page_index)
            in_toc = True
            continue
        if in_toc:
            break
    return toc_pages


# 收集多页重复出现的页眉页脚文本，用于后续去噪。
def collect_repeated_marginal_texts(doc, top_margin=80, bottom_margin=80, min_count=3):
    """
    Collect repeated text from page margins.

    Annual reports usually repeat report name, company name, and page number
    in headers/footers. Repeated short margin text is treated as noise.
    """
    candidates = Counter()
    for page in doc:
        height = page.rect.height
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text = block[:5]
            if y1 <= top_margin or y0 >= height - bottom_margin:
                key = compact_text(text)
                if key and len(key) <= 60 and not PAGE_NUMBER_RE.match(key):
                    candidates[key] += 1
    return {key for key, count in candidates.items() if count >= min_count}


# 估计页面正文字号中位数，用于判断标题和正文。
def get_page_font_median(page):
    """Estimate the normal body font size of one page."""
    sizes = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    sizes.append(float(span.get("size", 0)))
    return median(sizes) if sizes else 10.5


# 计算坐标矩形面积。
def rect_area(rect):
    """Calculate rectangle area from (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    return max(0, x1 - x0) * max(0, y1 - y0)


# 计算两个坐标矩形的重叠面积。
def overlap_area(a, b):
    """Calculate overlap area of two rectangles."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return rect_area((x0, y0, x1, y1))


# 判断一个文本框中心点是否落在目标坐标框内。
def is_inside(rect, bbox, tolerance=1.5):
    """Return True if the center of rect is inside bbox."""
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return bx0 - tolerance <= cx <= bx1 + tolerance and by0 - tolerance <= cy <= by1 + tolerance


# 判断文本块是否明显和表格区域重叠，避免表格文本重复抽取。
def block_overlaps_tables(block_bbox, table_bboxes, threshold=0.15):
    """Avoid extracting table text twice as both text block and table block."""
    area = rect_area(block_bbox)
    if area <= 0:
        return False
    for table_bbox in table_bboxes:
        if overlap_area(block_bbox, table_bbox) / area >= threshold:
            return True
    return False


# 用 pdfplumber 检测页面中的表格结构和单元格。
def find_table_structures(plumber_page):
    """
    Detect table structures with pdfplumber.

    We use pdfplumber for the table grid/bbox, then fill each cell with
    PyMuPDF words by coordinates. This is usually more reliable than directly
    trusting pdfplumber.extract_table() text.
    """
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
        "text_tolerance": 3,
    }
    tables = plumber_page.find_tables(table_settings=settings)
    if not tables:
        tables = plumber_page.find_tables()
    return tables


# 把 PyMuPDF 提取的 words 按阅读顺序拼成可读文本。
def words_to_text(words):
    """Convert PyMuPDF words into readable line-ordered text."""
    if not words:
        return ""
    words = sorted(words, key=lambda item: (round(item[1] / 3) * 3, item[0]))
    lines = []
    current_line = []
    current_y = None
    for word in words:
        x0, y0, x1, y1, text = word[:5]
        if current_y is None or abs(y0 - current_y) <= 4:
            current_line.append((x0, text))
            current_y = y0 if current_y is None else (current_y + y0) / 2
        else:
            lines.append(" ".join(t for _, t in sorted(current_line)))
            current_line = [(x0, text)]
            current_y = y0
    if current_line:
        lines.append(" ".join(t for _, t in sorted(current_line)))
    return normalize_text("\n".join(lines))


# 把检测到的表格转换为 Markdown 表格文本，便于后续 RAG 检索。
def table_to_markdown(table, page_words):
    """
    Fill a pdfplumber table grid with PyMuPDF words and return Markdown table.

    Markdown keeps the table easy to inspect and works well as RAG content.
    """
    rows = []
    for row in table.rows:
        row_cells = []
        for cell in row.cells:
            if cell is None:
                row_cells.append("")
                continue
            cell_words = [word for word in page_words if is_inside(word[:4], cell)]
            row_cells.append(words_to_text(cell_words).replace("\n", " "))
        if any(cell.strip() for cell in row_cells):
            rows.append(row_cells)

    if not rows:
        table_words = [word for word in page_words if is_inside(word[:4], table.bbox)]
        text = words_to_text(table_words)
        return text

    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    header = normalized_rows[0]
    separator = ["---"] * max_cols
    body = normalized_rows[1:]

    def md_row(row):
        escaped = [cell.replace("|", "\\|").strip() for cell in row]
        return "| " + " | ".join(escaped) + " |"

    return "\n".join([md_row(header), md_row(separator), *[md_row(row) for row in body]])


# 提取页面中的非表格文本块，并初步分类为标题或正文。
def extract_text_blocks(page, table_bboxes, repeated_marginals, top_margin=55, bottom_margin=55):
    """
    Extract non-table text blocks from one page.

    Text blocks overlapping table bboxes are skipped here because they will be
    reconstructed as table blocks later. Header/footer noise is removed before
    block classification.
    """
    page_height = page.rect.height
    page_font_median = get_page_font_median(page)
    blocks = []

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        if bbox[3] <= top_margin or bbox[1] >= page_height - bottom_margin:
            key = compact_text(extract_block_text(block))
            if key in repeated_marginals or PAGE_NUMBER_RE.match(key):
                continue
        text, font_sizes, bold_hits, span_count, filtered_bbox = extract_block_text_and_style(block, table_bboxes)
        if not text:
            continue
        key = compact_text(text)
        if key in repeated_marginals or PAGE_NUMBER_RE.match(key):
            continue

        max_font = max(font_sizes) if font_sizes else page_font_median
        avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else page_font_median
        block_type, title_level = classify_text_block(text, max_font, avg_font, page_font_median, bold_hits, span_count)
        blocks.append(
            {
                "kind": block_type,
                "bbox": filtered_bbox or bbox,
                "content": text,
                "title_level": title_level,
            }
        )
    return blocks


# 从 PyMuPDF 文本块中按行读取纯文本。
def extract_block_text(block):
    """Read text from a PyMuPDF dict block while preserving line order."""
    lines = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = "".join(spans).strip()
        if line_text:
            lines.append(line_text)
    return normalize_text("\n".join(lines))


# 判断某一行文字是否位于表格区域内。
def line_inside_tables(line_bbox, table_bboxes):
    """Return True if a text line belongs to a detected table area."""
    if not table_bboxes:
        return False
    return any(is_inside(line_bbox, table_bbox, tolerance=2.0) for table_bbox in table_bboxes)


# 把多个坐标框合并成一个覆盖范围。
def merge_bboxes(bboxes):
    """Merge multiple bboxes into one bbox."""
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


# 提取文本块内容、字体大小、加粗信息，同时剔除表格内文本行。
def extract_block_text_and_style(block, table_bboxes):
    """
    Extract text/style from a block after removing table lines.

    PyMuPDF sometimes merges the last row of a table and the following note
    into one large text block. If we dropped the whole block because it touched
    a table bbox, the note would disappear. Line-level filtering keeps notes
    such as "附注:" while still avoiding duplicate table text.
    """
    lines = []
    kept_bboxes = []
    font_sizes = []
    bold_hits = 0
    span_count = 0

    for line in block.get("lines", []):
        line_bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
        if line_inside_tables(line_bbox, table_bboxes):
            continue

        spans = line.get("spans", [])
        line_text = "".join(span.get("text", "") for span in spans).strip()
        if not line_text:
            continue

        lines.append(line_text)
        kept_bboxes.append(line_bbox)
        for span in spans:
            span_text = span.get("text", "").strip()
            if not span_text:
                continue
            font_sizes.append(float(span.get("size", 0)))
            font_name = span.get("font", "").lower()
            flags = int(span.get("flags", 0))
            if "bold" in font_name or flags & 16:
                bold_hits += 1
            span_count += 1

    return normalize_text("\n".join(lines)), font_sizes, bold_hits, span_count, merge_bboxes(kept_bboxes)


# 结合编号、长度、字号和加粗比例判断文本块是标题还是正文。
def classify_text_block(text, max_font, avg_font, page_font_median, bold_hits, span_count):
    """
    Classify one text block as title or text.

    Numbered headings are useful for section tracking, but annual reports also
    contain many numbered paragraphs. The checks below combine pattern, length,
    punctuation, font size, and bold ratio to reduce false titles.
    """
    single_line = "\n" not in text
    flat = normalize_text(text).replace("\n", " ")
    bold_ratio = bold_hits / span_count if span_count else 0
    for level, pattern in SECTION_PATTERNS:
        if not pattern.match(flat):
            continue
        if level == 1 and len(flat) <= 80:
            return "title", level
        numbered_title = (
            len(flat) <= 60
            and not flat.endswith(("。", "；", ";", "，", ","))
            and (max_font >= page_font_median + 0.8 or bold_ratio >= 0.4 or len(flat) <= 28)
        )
        if level > 1 and numbered_title:
            return "title", level

    looks_like_title = (
        single_line
        and 2 <= len(flat) <= 45
        and not flat.endswith(("。", "；", ";", "，", ","))
        and (max_font >= page_font_median + 1.2 or bold_ratio >= 0.5)
    )
    if looks_like_title and not re.search(r"\d{4}年\d{1,2}月\d{1,2}日", flat):
        return "title", infer_title_level(flat)
    return "text", None


# 根据标题编号规则推断章节层级。
def infer_title_level(text):
    """Infer heading level from numbering pattern."""
    for level, pattern in SECTION_PATTERNS:
        if pattern.match(text):
            return level
    return 6


# 遇到新标题时更新章节栈，维护当前章节路径。
def update_section_stack(section_stack, title, level):
    """
    Update the current section path.

    Example:
    第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析
    """
    level = level or infer_title_level(title)
    while section_stack and section_stack[-1]["level"] >= level:
        section_stack.pop()
    section_stack.append({"level": level, "title": title})
    return section_stack


# 把当前章节栈转换成 block metadata 里的 section 字符串。
def current_section_meta(section_stack):
    """Build section metadata from the current section stack."""
    path = [item["title"] for item in section_stack]
    return {
        "section": " > ".join(path),
    }


# 把内容、页码、文档元数据和章节信息封装成统一 block。
def make_block(content, block_type, page_number, doc_meta, section_stack):
    """Create the final JSON block with common metadata."""
    section_meta = current_section_meta(section_stack)
    return {
        "content": content,
        "meta": {
            "block_type": block_type,
            "page": page_number,
            "filename": doc_meta["filename"],
            "company_name": doc_meta["company_name"],
            "stock_code": doc_meta["stock_code"],
            "year": doc_meta["year"],
            **section_meta,
        },
    }


# 判断相邻正文元素是否应该合并成一个连续段落。
def should_merge_text_elements(previous, current, max_vertical_gap=8):
    """
    Decide whether two nearby text elements should become one text block.

    PyMuPDF may split one visual paragraph/list into several blocks. For RAG,
    keeping such nearby lines together is usually better than embedding tiny
    fragments. We only merge consecutive text elements with a small vertical
    gap, so tables and titles still form their own blocks.
    """
    if previous["kind"] != "text" or current["kind"] != "text":
        return False
    prev_bbox = previous["bbox"]
    curr_bbox = current["bbox"]
    vertical_gap = curr_bbox[1] - prev_bbox[3]
    if vertical_gap < 0:
        return False
    return vertical_gap <= max_vertical_gap


# 合并视觉上连续的正文块，减少过碎文本。
def merge_text_elements(elements):
    """
    Merge adjacent text elements that belong to the same visual paragraph.

    This fixes cases such as table notes where PyMuPDF creates:
    block 1: 附注 + first line of (1)
    block 2: continuation lines + (2)(3)
    but the desired output is one complete text block.
    """
    merged = []
    for element in elements:
        if merged and should_merge_text_elements(merged[-1], element):
            merged[-1]["content"] = normalize_text(merged[-1]["content"] + "\n" + element["content"])
            merged[-1]["bbox"] = merge_bboxes([merged[-1]["bbox"], element["bbox"]])
            continue
        merged.append(element)
    return merged


# 解析单个 PDF，输出 doc_meta 和有序 blocks。
def parse_pdf(pdf_path):
    """
    Parse one PDF into doc_meta and ordered blocks.

    The ordering strategy is simple and important: text blocks and table
    blocks are both represented by page coordinates, then sorted by y/x
    position so the JSON follows the original reading order.
    """
    doc = fitz.open(pdf_path)
    try:
        doc_meta = extract_doc_meta(pdf_path, doc)
        toc_pages = detect_toc_pages(doc)
        repeated_marginals = collect_repeated_marginal_texts(doc)
        blocks = []
        section_stack = []

        with pdfplumber.open(pdf_path) as plumber_pdf:
            for page_index, page in enumerate(doc):
                if page_index == 0 or page_index in toc_pages:
                    continue

                plumber_page = plumber_pdf.pages[page_index]
                tables = find_table_structures(plumber_page)
                table_bboxes = [table.bbox for table in tables]
                page_words = page.get_text("words")

                elements = extract_text_blocks(page, table_bboxes, repeated_marginals)
                for table in tables:
                    content = table_to_markdown(table, page_words)
                    if content:
                        elements.append(
                            {
                                "kind": "table",
                                "bbox": table.bbox,
                                "content": content,
                                "title_level": None,
                            }
                        )

                elements.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
                elements = merge_text_elements(elements)
                for element in elements:
                    content = normalize_text(element["content"])
                    if not content:
                        continue
                    page_number = page_index + 1
                    if element["kind"] == "title":
                        section_stack = update_section_stack(section_stack, content.replace("\n", " "), element["title_level"])
                        blocks.append(make_block(content, "title", page_number, doc_meta, section_stack))
                    elif element["kind"] == "table":
                        blocks.append(make_block(content, "table", page_number, doc_meta, section_stack))
                    else:
                        blocks.append(make_block(content, "text", page_number, doc_meta, section_stack))

        return {
            "doc_meta": {
                **doc_meta,
                "page_count": len(doc),
                "skipped_pages": {
                    "cover": [1],
                    "toc": [page + 1 for page in sorted(toc_pages)],
                },
            },
            "blocks": blocks,
        }
    finally:
        doc.close()


# 根据解析出的公司、股票代码和年份生成输出 JSON 路径。
def output_path_for(parsed, output_dir):
    """Build output JSON path from parsed metadata."""
    meta = parsed["doc_meta"]
    stock_code = meta.get("stock_code") or "unknown"
    year = meta.get("year") or "unknown"
    company = safe_filename_part(meta.get("company_name") or "unknown")
    return output_dir / f"{stock_code}_{year}_{company}.json"


# 批量解析输入目录下的所有 PDF 文件。
def parse_all(input_dir, output_dir, limit=None):
    """Parse all PDF files in input_dir and write one JSON per PDF."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Windows file matching is case-insensitive, so glob("*.pdf") and
    # glob("*.PDF") may return the same file twice. Filtering by suffix keeps
    # both .pdf and .PDF while avoiding duplicate parsing.
    pdf_paths = sorted(
        {
            path.resolve(): path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        }.values()
    )
    if limit:
        pdf_paths = pdf_paths[:limit]

    results = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{index}/{len(pdf_paths)}] Parsing {pdf_path.name} ...")
        parsed = parse_pdf(pdf_path)
        out_path = output_path_for(parsed, output_dir)
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {out_path} ({len(parsed['blocks'])} blocks)")
        results.append(out_path)
    return results


# 命令行入口：解析参数并启动批量 PDF 解析。
def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Parse annual report PDFs into structured JSON blocks.")
    parser.add_argument("--input-dir", default="data/raw_pdf", help="Directory containing PDF annual reports.")
    parser.add_argument("--output-dir", default="data/parsed_json", help="Directory for parsed JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Parse only the first N PDFs for quick checks.")
    args = parser.parse_args()

    parse_all(args.input_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
