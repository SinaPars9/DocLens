import pymupdf
import pdfplumber
import camelot
import os
import tempfile
import pandas as pd
import re
from langchain_core.documents import Document
from textlens import OCR

def block_overlaps_table(block, table_bbox, threshold=0.5):
    """Check whether a text block substantially overlaps a table."""
    x0, y0, x1, y1 = block[:4]
    tx0, ttop, tx1, tbottom = table_bbox

    block_area = max(0, x1 - x0) * max(0, y1 - y0)
    if block_area == 0:
        return False

    intersection_x0 = max(x0, tx0)
    intersection_y0 = max(y0, ttop)
    intersection_x1 = min(x1, tx1)
    intersection_y1 = min(y1, tbottom)

    intersection_width = max(0, intersection_x1 - intersection_x0)
    intersection_height = max(0, intersection_y1 - intersection_y0)

    intersection_area = intersection_width * intersection_height
    overlap_ratio = intersection_area / block_area

    return overlap_ratio >= threshold


def is_table_empty(text: str) -> bool:
    return not bool(re.search(r"[a-zA-Z0-9]", text))


def extract_page_content(pymupdf_page, plumber_page, page_number, pdf_path):
    """
    Extract text blocks and tables separately from a single page.
    Uses Camelot for tables (lattice), with pdfplumber as fallback.
    """
    camelot_tables_text = []
    table_bboxes = []

    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="lattice")  # type: ignore
        if len(tables) > 0:
            for table in tables:
                df = table.df
                if df.empty or len(df) < 2:
                    continue

                total_words = sum(
                    len(str(cell).split()) for row in df.values for cell in row if cell
                )
                total_cells = df.size
                if total_cells > 0 and (total_words / total_cells) > 6:
                    continue

                df.columns = df.iloc[0]
                df = df[1:]

                markdown_table = df.to_markdown(index=False)

                if is_table_empty(markdown_table):
                    continue

                camelot_tables_text.append(markdown_table)
                x0, y0, x1, y1 = table._bbox
                page_height = pymupdf_page.rect.height
                table_bboxes.append((x0, page_height - y1, x1, page_height - y0))
    except Exception as e:
        print(f"Camelot failed on page {page_number}: {e}")

    if not camelot_tables_text:
        table_objects = plumber_page.find_tables()
        for table_obj in table_objects:
            raw_table = table_obj.extract()
            if not raw_table or not raw_table[0] or len(raw_table[0]) < 2:
                continue

            total_words = sum(
                len(str(cell).split())
                for row in raw_table
                if row
                for cell in row
                if cell
            )
            total_cells = sum(len(row) for row in raw_table if row)
            if total_cells > 0 and (total_words / total_cells) > 6:
                continue

            df = pd.DataFrame(raw_table[1:], columns=raw_table[0])
            markdown_table = df.to_markdown(index=False)

            if is_table_empty(markdown_table):
                continue

            camelot_tables_text.append(markdown_table)
            table_bboxes.append(table_obj.bbox)

    text_blocks = pymupdf_page.get_text("blocks")
    text_elements = []

    for block in text_blocks:
        x0, y0, x1, y1 = block[:4]
        text = block[4].strip()

        if not text:
            continue

        belongs_to_table = any(
            block_overlaps_table(block, bbox) for bbox in table_bboxes
        )

        if belongs_to_table:
            continue

        text_elements.append((y0, x0, text))

    text_elements.sort(key=lambda element: (element[0], element[1]))
    text_content = "\n\n".join(element[2] for element in text_elements)

    return text_content, camelot_tables_text

def needs_ocr(pdf_path: str, min_chars_per_page: int = 50) -> bool:
    try:
        with pymupdf.open(pdf_path) as doc:
            n_pages = len(doc)
            if n_pages == 0:
                return False
            total = 0
            for page in doc:
                txt = page.get_text()
                if isinstance(txt, str):
                    total += len(txt.strip())
            return (total / n_pages) < min_chars_per_page
    except Exception as e:
        print(f"[needs_ocr] check failed: {e}")
        return False


def _page_to_text(page) -> str:
    for method_name in ("to_markdown", "to_text"):
        method = getattr(page, method_name, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, str) and value.strip():
                    return value.strip()
            except Exception:
                pass

    
    text = getattr(page, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    
    blocks = getattr(page, "blocks", None) or []
    parts = []
    for block in blocks:
        block_text = getattr(block, "text", None)
        if isinstance(block_text, str) and block_text.strip():
            parts.append(block_text.strip())
    return "\n\n".join(parts)


def _extract_with_textlens(pdf_path: str):
    ocr = OCR()
    result = ocr(pdf_path)

    pages = getattr(result, "pages", None) or []
    print(f"[TextLens] Pages: {len(pages)}, page_count: {getattr(result, 'page_count', '?')}")

    review_attr = getattr(result, "pages_needing_review", None)
    if callable(review_attr):
        try:
            review_attr = review_attr()
        except Exception as e:
            print(f"[TextLens] pages_needing_review() failed: {e}")
            review_attr = None
    if isinstance(review_attr, (list, tuple, set)) and len(review_attr) > 0:
        print(f"[TextLens] Pages needing review: {len(review_attr)}")

    documents = []
    for idx, page in enumerate(pages, start=1):
        content = _page_to_text(page)
        if not content:
            continue

        page_num = getattr(page, "number", idx)
        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = idx

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "type": "text",
                    "table_count": 0,
                    "ocr_applied": True,
                },
            )
        )

    if not documents:
        md = result.to_markdown()
        md = md.strip() if isinstance(md, str) else ""
        if md:
            print("[TextLens] No pages extracted, using whole markdown as fallback.")
            documents.append(
                Document(
                    page_content=md,
                    metadata={
                        "source": pdf_path,
                        "page": 1,
                        "type": "text",
                        "table_count": 0,
                        "ocr_applied": True,
                    },
                )
            )

    return documents

def load_pdf_documents(pdf_path):
    """
    Create LangChain Documents from a PDF.
    - One Document per page of ordinary text (metadata: type="text").
    - One Document per detected table   (metadata: type="table").
    """
    if needs_ocr(pdf_path):
        print(f"[TextLens] Scanned PDF detected: {os.path.basename(pdf_path)}")
        try:
            docs = _extract_with_textlens(pdf_path)
            if docs:
                print(f"[TextLens] Extracted {len(docs)} pages.")
                return docs
            print("[TextLens] Empty output, falling back to native extraction.")
        except Exception as e:
            print(f"[TextLens] Failed, falling back: {e}")

    documents = []

    with pymupdf.open(pdf_path) as pdf_text, pdfplumber.open(pdf_path) as pdf_tables:

        if len(pdf_text) != len(pdf_tables.pages):
            raise ValueError("The PDF readers returned different page counts.")

        for page_index in range(len(pdf_text)):
            page_number = page_index + 1
            pymupdf_page = pdf_text[page_index]
            plumber_page = pdf_tables.pages[page_index]

            text_content, tables = extract_page_content(
                pymupdf_page, plumber_page, page_number, pdf_path
            )

            # ---- Document for the page's ordinary text ----
            if text_content.strip():
                documents.append(
                    Document(
                        page_content=text_content,
                        metadata={
                            "source": pdf_path,
                            "page": page_number,
                            "type": "text",
                            "table_count": len(tables),
                        },
                    )
                )

            # ---- One Document per table ----
            for table_index, table_text in enumerate(tables):
                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "source": pdf_path,
                            "page": page_number,
                            "type": "table",
                            "table_index": table_index,
                        },
                    )
                )

    return documents
