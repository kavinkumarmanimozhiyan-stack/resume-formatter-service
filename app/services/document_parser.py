import fitz
import os


def extract_text_from_document(file_path: str) -> str:
    """
    Extracts text content from a PDF or Word document.

    Args:
        file_path: Absolute path to the document file

    Returns:
        Extracted text content as a string
    """
    print(f"[DOCUMENT_PARSER] extract_text_from_document called: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return _extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return _extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    print(f"[DOCUMENT_PARSER] Extracting text from PDF: {file_path}")
    doc = fitz.open(file_path)
    print(f"[DOCUMENT_PARSER] PDF opened, page count: {doc.page_count}")
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        text += page_text
        print(f"[DOCUMENT_PARSER] Page {page_num}: {len(page_text)} chars")
    doc.close()
    print(f"[DOCUMENT_PARSER] Total text extracted from PDF: {len(text)} chars")
    return text.strip()


def _extract_text_from_docx(file_path: str) -> str:
    """Extract text from Word .docx using the shared docx_parser (includes tables)."""
    from app.services.docx_parser import extract_text_from_docx as extract_docx_text

    return extract_docx_text(file_path)


def extract_format_from_document(file_path: str) -> dict:
    """
    Extracts visual format information from a PDF or Word document.

    Args:
        file_path: Absolute path to the document file

    Returns:
        Dict with keys: page_count, layout, typography, color_palette, sections, pages
    """
    print(f"[DOCUMENT_PARSER] extract_format_from_document called: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return _extract_format_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return _extract_format_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def _extract_format_from_pdf(file_path: str) -> dict:
    """Extract format information from PDF using PyMuPDF"""
    print(f"[DOCUMENT_PARSER] Extracting format from PDF: {file_path}")
    doc = fitz.open(file_path)
    page_count = doc.page_count
    print(f"[DOCUMENT_PARSER] PDF opened for format extraction, page count: {page_count}")

    all_font_sizes = []
    all_colors = set()
    sections = []
    pages_data = []

    # Analyze each page
    for page_num, page in enumerate(doc):
        print(f"[DOCUMENT_PARSER] Analyzing page {page_num}")
        page_dict = page.get_text("dict")
        page_data = {
            "page_num": page_num,
            "width": page.rect.width,
            "height": page.rect.height,
            "blocks": []
        }

        # Column detection
        page_midpoint = page.rect.width / 2
        left_blocks = 0
        right_blocks = 0
        left_max_x1 = 0
        right_min_x0 = float('inf')

        block_count = 0
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue

            block_count += 1
            block_center_x = (block["bbox"][0] + block["bbox"][2]) / 2

            if block_center_x < page_midpoint:
                left_blocks += 1
                left_max_x1 = max(left_max_x1, block["bbox"][2])
            else:
                right_blocks += 1
                right_min_x0 = min(right_min_x0, block["bbox"][0])

            block_data = {"spans": []}

            span_count = 0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_count += 1
                    # Extract span details
                    span_data = {
                        "text": span.get("text", ""),
                        "font": span.get("font", ""),
                        "size": span.get("size", 0),
                        "flags": span.get("flags", 0),
                        "color": span.get("color", 0),
                        "bbox": span.get("bbox", [0, 0, 0])
                    }

                    # Extract font size
                    font_size = span_data["size"]
                    if font_size > 0:
                        all_font_sizes.append(font_size)

                    # Extract color (convert int to hex)
                    color_int = span_data["color"]
                    color_hex = f"#{color_int:06x}"
                    all_colors.add(color_hex)

                    # Check for bold (flags & 16) or "bold" in font name
                    is_bold = (span_data["flags"] & 16) or "bold" in span_data["font"].lower()

                    # Check for italic (flags & 2)
                    is_italic = span_data["flags"] & 2

                    span_data["is_bold"] = is_bold
                    span_data["is_italic"] = is_italic
                    span_data["color_hex"] = color_hex

                    block_data["spans"].append(span_data)

                    # Detect section headers: short text, bold, or large font
                    text = span_data["text"].strip()
                    if text and len(text) < 30 and (is_bold or font_size >= 12):
                        sections.append(text)

            page_data["blocks"].append(block_data)

        print(f"[DOCUMENT_PARSER] Page {page_num}: {block_count} blocks, {span_count} spans")

        pages_data.append(page_data)

        # Determine column layout for this page
        is_two_column = left_blocks > 2 and right_blocks > 2
        column_boundary = None
        if is_two_column and left_max_x1 > 0 and right_min_x0 < float('inf'):
            column_boundary = (left_max_x1 + right_min_x0) / 2

        page_data["layout"] = {
            "is_two_column": is_two_column,
            "column_boundary": column_boundary
        }
        print(f"[DOCUMENT_PARSER] Page {page_num} layout: two_column={is_two_column}, left_blocks={left_blocks}, right_blocks={right_blocks}")

    doc.close()

    # Typography summary
    unique_font_sizes = sorted(set(all_font_sizes), reverse=True)
    name_size = unique_font_sizes[0] if unique_font_sizes else 14
    heading_size = unique_font_sizes[1] if len(unique_font_sizes) > 1 else 12
    body_size = unique_font_sizes[-1] if unique_font_sizes else 10

    print(f"[DOCUMENT_PARSER] Typography: name_size={name_size}, heading_size={heading_size}, body_size={body_size}")
    print(f"[DOCUMENT_PARSER] Found {len(all_colors)} unique colors, {len(sections)} section headers")

    # Overall layout (use first page as representative)
    overall_layout = pages_data[0]["layout"] if pages_data else {
        "is_two_column": False,
        "column_boundary": None
    }

    result = {
        "page_count": page_count,
        "layout": overall_layout,
        "typography": {
            "name_size": name_size,
            "heading_size": heading_size,
            "body_size": body_size
        },
        "color_palette": sorted(list(all_colors)),
        "sections": sections,
        "pages": pages_data
    }
    print(f"[DOCUMENT_PARSER] PDF format extraction complete")
    return result


def _extract_format_from_docx(file_path: str) -> dict:
    """Extract format information from .docx using docx_parser design tokens."""
    from app.services.docx_parser import extract_design_tokens

    print(f"[DOCUMENT_PARSER] Extracting format from Word document: {file_path}")
    tokens = extract_design_tokens(file_path)
    sidebar = tokens.get("sidebar", {}) or {}
    result = {
        "page_count": 1,
        "layout": {
            "is_two_column": sidebar.get("has_sidebar", False),
            "column_boundary": sidebar.get("width_px"),
        },
        "sidebar": sidebar,
        "typography": tokens.get("typography", {}),
        "color_palette": tokens.get("color_palette", []),
        "sections": tokens.get("sections", []),
        "pages": [],
    }
    print(f"[DOCUMENT_PARSER] Word format extraction complete")
    return result
