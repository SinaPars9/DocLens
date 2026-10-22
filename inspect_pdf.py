import pymupdf
import pdfplumber

from langchain_core.documents import Document


def table_to_text(table, page_number):
    """Convert a detected table into generic structured text."""

    if not table or len(table) < 2:
        return ""

    headers = [str(header).strip() if header is not None else "" for header in table[0]]

    lines = [f"Table (Page {page_number})"]

    for row in table[1:]:
        if not row:
            continue

        if all(value is None or not str(value).strip() for value in row):
            continue

        item_name = (
            str(row[0]).strip()
            if row[0] is not None and str(row[0]).strip()
            else "Unnamed item"
        )

        lines.append(f"\n{item_name}:")

        for header, value in zip(headers[1:], row[1:]):
            if value is None or not str(value).strip():
                continue

            label = header or "Value"
            lines.append(f"{label}: {str(value).strip()}")

    return "\n".join(lines)


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


def extract_page_content(pymupdf_page, plumber_page, page_number):
    """
    Extract text blocks and tables separately from a single page.

    Returns:
        text_content: string containing only the ordinary text of the page.
        tables: list of strings, one per detected table.
    """

    table_objects = plumber_page.find_tables()
    table_bboxes = []
    tables = []

    for table_obj in table_objects:
        table_bboxes.append(table_obj.bbox)

        raw_table = table_obj.extract()
        converted_text = table_to_text(raw_table, page_number)

        if converted_text:
            tables.append(converted_text)

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

    return text_content, tables


def load_pdf_documents(pdf_path):
    """
    Create LangChain Documents from a PDF.

    - One Document per page of ordinary text (metadata: type="text").
    - One Document per detected table   (metadata: type="table").
    """

    documents = []

    with pymupdf.open(pdf_path) as pdf_text, pdfplumber.open(pdf_path) as pdf_tables:

        if len(pdf_text) != len(pdf_tables.pages):
            raise ValueError("The PDF readers returned different page counts.")

        for page_index, pymupdf_page in enumerate(pdf_text):
            page_number = page_index + 1
            plumber_page = pdf_tables.pages[page_index]

            text_content, tables = extract_page_content(
                pymupdf_page, plumber_page, page_number
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
