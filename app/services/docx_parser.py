"""
docx_parser.py
─────────────────────────────────────────────────────────────────────────────
Extract plain text and design tokens from .docx (WordprocessingML) files.

The public shape of extract_design_tokens() matches pdf_parser.extract_design_tokens
so gemini_service can treat PDF and DOCX format sources the same way.

LOGO SUPPORT:
  extract_images_from_docx() pulls embedded raster images out of the .docx
  zip via python-docx relationship parts, with paragraph-position context.
  classify_logo_image() picks the most likely "company logo" candidate
  (small, roughly square, near the header). The winning image's raw bytes
  are base64-encoded into tokens["logo"] and are never redrawn/interpreted
  by the LLM — gemini_service splices the exact bytes back into the final
  HTML via a placeholder substitution.

LOGO PLACEMENT FIX (this version):
  .docx has no page geometry, so we can't compute x/y percentages the way
  pdf_parser does. But we DO know the image's paragraph index, whether that
  paragraph sits inside the sidebar table column vs the main column (when a
  sidebar table exists), and the nearby paragraph text. Previously none of
  this made it past classify_logo_image() into the returned token — only
  bytes_b64/ext survived. Now tokens["logo"] also includes:
    zone            — "sidebar" | "header" | "main"
    para_index      — 0-based paragraph order
    total_paragraphs
    near_text       — text of the paragraph the image is anchored in
  so gemini_service can build a placement-aware instruction instead of
  always defaulting to "top of header".
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import re
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.ns import qn
from docx.shared import Length
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract all visible text in document order: body paragraphs, table cells
    (row-major), including nested tables.
    """
    print(f"[DOCX_PARSER] extract_text_from_docx: {file_path}")
    doc = Document(file_path)
    parts: List[str] = []
    for para in _iter_paragraphs(doc):
        t = para.text.strip()
        if t:
            parts.append(para.text)
    out = "\n".join(parts).strip()
    print(f"[DOCX_PARSER] total text: {len(out)} chars")
    return out


def extract_images_from_docx(file_path: str) -> List[dict]:
    """
    Return embedded raster images with rough document-position context.

    Each item:
        {
            "bytes": <raw image bytes>,
            "content_type": "image/png" | "image/jpeg" | ...,
            "ext": "png" | "jpeg" | ...,
            "para_index": <int, 0-based paragraph order in the doc>,
            "near_text": <text of the paragraph the image is anchored in>,
            "width_emu": <int or None, drawing extent width in EMUs>,
            "height_emu": <int or None, drawing extent height in EMUs>,
            "in_table_cell": <bool, True if the paragraph lives inside a table cell>,
            "cell_col_index": <int or None, 0-based column index within its row,
                                only set when in_table_cell is True>,
        }

    `para_index` / `near_text` / `in_table_cell` / `cell_col_index` let the
    caller reason about position (e.g. "this image sits in one of the first
    few paragraphs" -> header logo, vs "this image sits inside column 0 of
    a 2-column sidebar table" -> sidebar logo) without needing full layout
    geometry, which .docx doesn't expose the way a rasterized PDF page does.
    """
    print(f"[DOCX_PARSER] extract_images_from_docx: {file_path}")
    doc = Document(file_path)
    rels = doc.part.rels

    images: List[dict] = []
    para_index = 0
    for para, cell_ctx in _iter_paragraphs_with_context(doc):
        blips = para._p.findall(".//" + qn("a:blip"))
        if not blips:
            para_index += 1
            continue

        # Drawing extent (cx/cy, in EMUs) usually lives on the sibling
        # <wp:extent> element inside the same <w:drawing>. Best-effort only.
        extents = para._p.findall(".//" + qn("wp:extent"))

        for i, blip in enumerate(blips):
            rId = blip.get(qn("r:embed"))
            if not rId or rId not in rels:
                continue
            part = rels[rId].target_part
            content_type = getattr(part, "content_type", "") or ""
            if "image" not in content_type:
                continue

            width_emu = height_emu = None
            if i < len(extents):
                try:
                    width_emu = int(extents[i].get("cx"))
                    height_emu = int(extents[i].get("cy"))
                except (TypeError, ValueError):
                    pass

            images.append({
                "bytes": part.blob,
                "content_type": content_type,
                "ext": content_type.split("/")[-1].replace("jpeg", "jpg"),
                "para_index": para_index,
                "near_text": para.text.strip(),
                "width_emu": width_emu,
                "height_emu": height_emu,
                "in_table_cell": cell_ctx is not None,
                "cell_col_index": cell_ctx[0] if cell_ctx else None,
            })

        para_index += 1

    print(f"[DOCX_PARSER] extracted {len(images)} embedded image(s)")
    return images


def classify_logo_image(
    images: List[dict],
    total_paragraphs: int,
    header_zone_paragraphs: int = 8,
) -> Optional[dict]:
    """
    Pick the most likely "company logo" image from a set of embedded
    images extracted from a .docx template.

    Heuristic (no page geometry available for .docx, so this is
    paragraph-position + aspect-ratio based):
      1. Prefer images that are roughly square (logos usually are;
         a wide banner or a full headshot photo usually isn't).
      2. Prefer images that are small relative to typical inline images
         (EMU width under ~1.2 inch == 1,097,280 EMU) when size info is
         available. If size info is missing, don't filter on it.
      3. Prefer images anchored in the header zone (first N paragraphs)
         — this is where a template's brand mark / logo mark usually
         lives, as opposed to a full-bleed background image elsewhere.
      4. If nothing matches the header zone, fall back to the smallest
         roughly-square image anywhere in the document (covers templates
         that put a logo mark next to the job-entry area instead).

    Returns None if no image looks like a plausible logo.
    """
    if not images:
        return None

    MAX_LOGO_EMU = 1_097_280  # ~1.2 inch — generous upper bound for a logo mark

    def aspect_ok(im: dict) -> bool:
        w, h = im.get("width_emu"), im.get("height_emu")
        if not w or not h:
            return True  # unknown size — don't exclude, just don't prioritize
        ratio = w / h if h else 0
        return 0.4 <= ratio <= 2.5

    def is_small(im: dict) -> bool:
        w, h = im.get("width_emu"), im.get("height_emu")
        if not w or not h:
            return False  # unknown size — treat as "not confirmed small"
        return w <= MAX_LOGO_EMU and h <= MAX_LOGO_EMU

    candidates = [im for im in images if aspect_ok(im)]
    if not candidates:
        candidates = images  # don't give up entirely on weird aspect ratios

    header_candidates = [
        im for im in candidates if im["para_index"] < header_zone_paragraphs
    ]

    def sort_key(im: dict):
        # Prefer confirmed-small images, then smaller images, then earlier
        # in the document.
        w = im.get("width_emu") or MAX_LOGO_EMU * 4
        h = im.get("height_emu") or MAX_LOGO_EMU * 4
        return (0 if is_small(im) else 1, w * h, im["para_index"])

    pool = header_candidates if header_candidates else candidates
    pool = sorted(pool, key=sort_key)

    winner = pool[0] if pool else None
    if winner:
        print(
            f"[DOCX_PARSER] classify_logo_image: selected image at "
            f"para_index={winner['para_index']} near_text={winner['near_text']!r} "
            f"size=({winner.get('width_emu')}, {winner.get('height_emu')}) "
            f"in_table_cell={winner.get('in_table_cell')} cell_col_index={winner.get('cell_col_index')}"
        )
    else:
        print("[DOCX_PARSER] classify_logo_image: no suitable logo candidate found")
    return winner


def _classify_logo_zone_docx(
    logo_image: dict,
    sidebar: dict,
    header_zone_paragraphs: int = 8,
) -> str:
    """
    Decide which layout region the logo sits in, using paragraph position
    and (when the doc uses a sidebar table) which column the image's
    paragraph lives in.

    Priority:
      1. If the doc has a detected sidebar table AND the image is inside a
         table cell AND that cell's column index matches the sidebar's
         side (0 == left column, last column == right column) -> "sidebar"
      2. Else if it's within the first `header_zone_paragraphs` paragraphs
         -> "header"
      3. Else -> "main"
    """
    if sidebar.get("has_sidebar") and logo_image.get("in_table_cell"):
        col_index = logo_image.get("cell_col_index")
        side = sidebar.get("side")
        if side == "left" and col_index == 0:
            return "sidebar"
        if side == "right" and col_index is not None and col_index > 0:
            return "sidebar"

    if logo_image["para_index"] < header_zone_paragraphs:
        return "header"

    return "main"


def extract_design_tokens(file_path: str) -> dict:
    """
    Walk the .docx structure and infer the same design token dict as
    pdf_parser.extract_design_tokens() so Gemini merge / HTML pass stay aligned.
    """
    print(f"[DOCX_PARSER] extract_design_tokens: {file_path}")
    doc = Document(file_path)

    sec = doc.sections[0] if doc.sections else None
    page_w = float(sec.page_width.pt) if sec and sec.page_width else 595.0
    page_h = float(sec.page_height.pt) if sec and sec.page_height else 842.0
    print(f"[DOCX_PARSER] page size (from section): {page_w:.1f} x {page_h:.1f} pt")

    spans: List[dict] = []
    para_index = 0
    for para in _iter_paragraphs(doc):
        in_header = para_index < 8
        for s in _paragraph_to_spans(para, in_header_zone=in_header):
            spans.append(s)
        para_index += 1

    total_paragraphs = para_index

    print(f"[DOCX_PARSER] collected {len(spans)} spans")

    sizes = sorted([s["size"] for s in spans if s["size"] > 0])
    body_size = _median(sizes) if sizes else 10.0
    name_size = max(sizes) if sizes else 24.0
    heading_size = _percentile(sizes, 85) if sizes else 12.0
    print(f"[DOCX_PARSER] sizes — body:{body_size:.1f} heading:{heading_size:.1f} name:{name_size:.1f}")

    sections = _extract_sections(spans, body_size)
    print(f"[DOCX_PARSER] sections: {sections}")

    color_palette = _extract_palette(spans)
    accent_color = _pick_accent(color_palette)
    print(f"[DOCX_PARSER] accent: {accent_color}  palette: {color_palette}")

    sidebar = _detect_sidebar_from_tables(doc, page_w)
    print(f"[DOCX_PARSER] sidebar (from tables): {sidebar}")

    header_info = _analyze_header_docx(spans, name_size, sidebar)
    print(f"[DOCX_PARSER] header: {header_info}")

    heading_info = _analyze_heading_style_docx(spans, body_size, heading_size)
    print(f"[DOCX_PARSER] heading style: {heading_info}")

    body_colors = [s["color_hex"] for s in spans if abs(s["size"] - body_size) < 1.5]
    body_color = _mode(body_colors) if body_colors else "#333333"

    bullet_char = _detect_bullet_docx(spans)
    print(f"[DOCX_PARSER] bullet: {bullet_char!r}")

    font_family = _extract_dominant_font(spans, body_size)
    print(f"[DOCX_PARSER] font family: {font_family}")

    # ── Logo extraction (WITH zone inference, so placement can be exact) ────
    # Never let extraction failure break the whole pipeline — a missing
    # logo is a silent no-op (the placeholder tag just doesn't get
    # inserted downstream), never a hard error.
    logo_token: dict = {"present": False, "bytes_b64": None, "ext": None}
    try:
        images = extract_images_from_docx(file_path)
        logo_image = classify_logo_image(images, total_paragraphs)
        if logo_image:
            zone = _classify_logo_zone_docx(logo_image, sidebar)
            logo_token = {
                "present": True,
                "bytes_b64": base64.b64encode(logo_image["bytes"]).decode("ascii"),
                "ext": logo_image["ext"],
                "near_text": logo_image.get("near_text", ""),
                "para_index": logo_image["para_index"],
                "total_paragraphs": total_paragraphs,
                "zone": zone,
            }
            print(f"[DOCX_PARSER] logo zone inferred: {zone}")
    except Exception as e:
        print(f"[DOCX_PARSER] WARNING: logo extraction failed: {e}")
    print(f"[DOCX_PARSER] logo: present={logo_token['present']}")

    if sidebar.get("has_sidebar"):
        layout_type = f"sidebar-{sidebar['side']}"
    else:
        layout_type = "single-column"

    return {
        "layout_type": layout_type,
        "sidebar": sidebar,
        "header": header_info,
        "section_heading": heading_info,
        "body": {"color": body_color, "size_pt": body_size},
        "accent_color": accent_color,
        "color_palette": color_palette,
        "sections": sections,
        "section_order": sections,
        "bullet_char": bullet_char,
        "font_family": font_family,
        "logo": logo_token,
        "typography": {
            "name_size": name_size,
            "heading_size": heading_size,
            "body_size": body_size,
            "font_family": font_family,
        },
        "page_size": {"width_pt": page_w, "height_pt": page_h},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Document walk
# ─────────────────────────────────────────────────────────────────────────────


def _iter_paragraphs(document: DocumentObject) -> Iterator[Paragraph]:
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield from _iter_paragraphs_in_table(Table(child, document))


def _iter_paragraphs_in_table(table: Table) -> Iterator[Paragraph]:
    for row in table.rows:
        for cell in row.cells:
            for child in cell._tc.iterchildren():
                if child.tag == qn("w:p"):
                    yield Paragraph(child, cell)
                elif child.tag == qn("w:tbl"):
                    yield from _iter_paragraphs_in_table(Table(child, cell))


def _iter_paragraphs_with_context(
    document: DocumentObject,
) -> Iterator[tuple[Paragraph, Optional[tuple[int, int]]]]:
    """
    Same traversal as _iter_paragraphs, but also yields, for paragraphs
    inside a table cell, a (col_index, row_index) tuple identifying which
    cell the paragraph lives in. Yields None as the context for paragraphs
    that are NOT inside any table. Used only by extract_images_from_docx so
    we can tell whether an embedded image sits in a sidebar column.
    """
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document), None
        elif child.tag == qn("w:tbl"):
            yield from _iter_paragraphs_in_table_with_context(Table(child, document))


def _iter_paragraphs_in_table_with_context(
    table: Table,
) -> Iterator[tuple[Paragraph, Optional[tuple[int, int]]]]:
    for row_index, row in enumerate(table.rows):
        for col_index, cell in enumerate(row.cells):
            for child in cell._tc.iterchildren():
                if child.tag == qn("w:p"):
                    yield Paragraph(child, cell), (col_index, row_index)
                elif child.tag == qn("w:tbl"):
                    yield from _iter_paragraphs_in_table_with_context(Table(child, cell))


# ─────────────────────────────────────────────────────────────────────────────
#  Runs → spans
# ─────────────────────────────────────────────────────────────────────────────


def _paragraph_to_spans(para: Paragraph, in_header_zone: bool) -> List[dict]:
    spans: List[dict] = []
    p_style_font = para.style.font if para.style and para.style.font else None

    if not para.runs:
        t = para.text.strip()
        if not t:
            return []
        base_sz = _length_to_pt(p_style_font.size) if p_style_font and p_style_font.size else None
        spans.append(
            _make_span(
                text=para.text,
                size_pt=base_sz,
                bold=None,
                italic=None,
                color_hex=None,
                font_name=None,
                p_style_font=p_style_font,
                in_header_zone=in_header_zone,
            )
        )
        return [s for s in spans if s]

    for run in para.runs:
        text = run.text
        if not text or not text.strip():
            continue
        s = _make_span(
            text=text,
            size_pt=_length_to_pt(run.font.size),
            bold=run.bold,
            italic=run.italic,
            color_hex=_run_color_hex(run),
            font_name=run.font.name,
            p_style_font=p_style_font,
            in_header_zone=in_header_zone,
        )
        if s:
            spans.append(s)
    return spans


def _make_span(
    *,
    text: str,
    size_pt: Optional[float],
    bold: Optional[bool],
    italic: Optional[bool],
    color_hex: Optional[str],
    font_name: Optional[str],
    p_style_font: Any,
    in_header_zone: bool,
) -> Optional[dict]:
    t = text.strip()
    if not t:
        return None

    sz = size_pt
    if sz is None or sz <= 0:
        sz = _length_to_pt(p_style_font.size) if p_style_font and p_style_font.size else 11.0

    if bold is None and p_style_font is not None:
        bold = p_style_font.bold
    if italic is None and p_style_font is not None:
        italic = p_style_font.italic
    if font_name is None and p_style_font is not None:
        font_name = p_style_font.name

    if color_hex is None and p_style_font is not None and p_style_font.color and p_style_font.color.rgb:
        color_hex = _rgb_to_hex(p_style_font.color.rgb)

    if color_hex is None:
        color_hex = "#333333"
    color_int = _hex_to_int(color_hex)

    is_bold = bool(bold)
    is_italic = bool(italic)
    if not is_bold and p_style_font and p_style_font.bold:
        is_bold = True

    return {
        "text": t,
        "size": float(sz),
        "is_bold": is_bold,
        "is_italic": is_italic,
        "color_hex": color_hex,
        "color_int": color_int,
        "font": (font_name or "Calibri"),
        "in_header_zone": in_header_zone,
    }


def _length_to_pt(length: Optional[Length]) -> Optional[float]:
    if length is None:
        return None
    try:
        return float(length.pt)
    except Exception:
        return None


def _run_color_hex(run) -> Optional[str]:
    try:
        c = run.font.color
        if c is None or c.rgb is None:
            return None
        return _rgb_to_hex(c.rgb)
    except Exception:
        return None


def _rgb_to_hex(rgb) -> str:
    try:
        r, g, b = rgb
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    except Exception:
        return "#333333"


def _hex_to_int(h: str) -> int:
    h = h.strip().lstrip("#")
    if len(h) == 6:
        return int(h, 16)
    return 0x333333


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar (two-column tables)
# ─────────────────────────────────────────────────────────────────────────────


def _cell_shading_hex(cell: _Cell) -> Optional[str]:
    try:
        tc = cell._tc
        tcPr = tc.tcPr
        if tcPr is None:
            return None
        shd = tcPr.find(qn("w:shd"))
        if shd is None:
            return None
        fill = shd.get(qn("w:fill"))
        if not fill or fill.lower() in ("auto", "ffffff", "ffffffff"):
            return None
        if len(fill) == 6:
            return f"#{fill.upper()}"
        if len(fill) == 8 and fill.upper().startswith("FF"):
            return f"#{fill[2:8].upper()}"
        return None
    except Exception:
        return None


def _first_body_table(doc: DocumentObject) -> Optional[Table]:
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:tbl"):
            return Table(child, doc)
    return None


def _detect_sidebar_from_tables(doc: DocumentObject, page_w: float) -> dict:
    """
    Many resume .docx use a 2-column table: colored narrow column = sidebar.
    """
    tbl = _first_body_table(doc)
    if tbl is None or len(tbl.rows) < 1:
        return {"has_sidebar": False}

    row0 = tbl.rows[0]
    cells = row0.cells
    if len(cells) < 2:
        return {"has_sidebar": False}

    c0, c1 = cells[0], cells[1]
    fill0 = _cell_shading_hex(c0)
    fill1 = _cell_shading_hex(c1)

    w0_emu = c0.width
    w1_emu = c1.width
    w0_pt = w0_emu.pt if w0_emu is not None else None
    w1_pt = w1_emu.pt if w1_emu is not None else None

    total_w = None
    if w0_pt is not None and w1_pt is not None:
        total_w = w0_pt + w1_pt

    left_ratio = None
    if total_w and total_w > 0 and w0_pt is not None:
        left_ratio = w0_pt / total_w

    has_fill_contrast = bool(fill0 and not fill1) or bool(fill1 and not fill0)
    narrow_left = left_ratio is not None and left_ratio <= 0.42

    if not (has_fill_contrast or narrow_left):
        return {"has_sidebar": False}

    # Decide which column is the sidebar band
    if fill0 and not fill1:
        side, bg = "left", fill0
        width_pt = int(w0_pt) if w0_pt else int(page_w * 0.28)
    elif fill1 and not fill0:
        side, bg = "right", fill1
        width_pt = int(w1_pt) if w1_pt else int(page_w * 0.28)
    elif narrow_left:
        side, bg = "left", fill0 or "#f0f4f8"
        width_pt = int(w0_pt) if w0_pt else int(page_w * 0.28)
    else:
        side, bg = "right", fill1 or "#f0f4f8"
        width_pt = int(w1_pt) if w1_pt else int(page_w * 0.28)

    width_pct = round((width_pt / page_w) * 100, 1) if page_w else 30.0

    return {
        "has_sidebar": True,
        "side": side,
        "bg_color": bg,
        "width_px": width_pt,
        "width_pct": width_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis (mirrors pdf_parser heuristics where possible)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_sections(spans: List[dict], body_size: float) -> List[str]:
    heading_threshold = body_size * 1.25
    sections: List[str] = []
    seen: set[str] = set()

    for s in spans:
        text = s["text"].strip()
        size = s["size"]
        upper = text.isupper()
        word_count = len(text.split())

        if len(text) > 40:
            continue
        if any(c.isdigit() for c in text[:4]):
            continue
        if text.lower() in {
            "resume",
            "curriculum vitae",
            "cv",
            "page",
            "references available upon request",
        }:
            continue

        is_heading = (
            (upper and len(text) > 2 and word_count <= 5)
            or (size >= heading_threshold and s["is_bold"] and word_count <= 5)
        )

        key = text.lower()
        if is_heading and key not in seen:
            sections.append(text)
            seen.add(key)

    return sections


def _analyze_header_docx(spans: List[dict], name_size: float, sidebar: dict) -> dict:
    header_spans = [s for s in spans if s.get("in_header_zone")]
    name_spans = [s for s in header_spans if abs(s["size"] - name_size) < 2.5]
    name_color = name_spans[0]["color_hex"] if name_spans else "#000000"
    name_size_actual = name_spans[0]["size"] if name_spans else name_size

    bg_color = None
    if sidebar.get("has_sidebar") and sidebar.get("bg_color"):
        if name_spans:
            nc = name_spans[0]["color_int"]
            r_n = (nc >> 16) & 0xFF
            g_n = (nc >> 8) & 0xFF
            b_n = nc & 0xFF
            if r_n > 180 and g_n > 180 and b_n > 180:
                bg_color = sidebar["bg_color"]

    return {
        "bg_color": bg_color,
        "name_color": name_color,
        "name_size_pt": name_size_actual,
    }


def _analyze_heading_style_docx(spans: List[dict], body_size: float, heading_size: float) -> dict:
    threshold = body_size * 1.2
    heading_spans = [
        s
        for s in spans
        if (s["size"] >= threshold or s["text"].isupper())
        and len(s["text"]) < 40
        and not any(c.isdigit() for c in s["text"][:4])
        and len(s["text"].split()) <= 5
    ]
    if not heading_spans:
        return {
            "color": "#333333",
            "bg_color": None,
            "uppercase": False,
            "bold": True,
            "size_pt": heading_size,
        }
    colors = [s["color_hex"] for s in heading_spans]
    uppers = [s["text"].isupper() for s in heading_spans]
    bolds = [s["is_bold"] for s in heading_spans]
    return {
        "color": _mode(colors),
        "bg_color": None,
        "uppercase": sum(uppers) > len(uppers) / 2,
        "bold": sum(bolds) > len(bolds) / 2,
        "size_pt": heading_size,
    }


def _extract_palette(spans: List[dict]) -> List[str]:
    freq: Dict[str, int] = {}
    for s in spans:
        c = s["color_int"]
        if c < 0x222222 or c > 0xDDDDDD:
            continue
        h = s["color_hex"]
        freq[h] = freq.get(h, 0) + 1
    return sorted(freq, key=lambda h: -freq[h])[:8]


def _pick_accent(palette: List[str]) -> str:
    if not palette:
        return "#2c3e50"

    def saturation(hx: str) -> float:
        r = int(hx[1:3], 16) / 255
        g = int(hx[3:5], 16) / 255
        b = int(hx[5:7], 16) / 255
        mx, mn = max(r, g, b), min(r, g, b)
        return 0.0 if mx == 0 else (mx - mn) / mx

    return max(palette, key=saturation)


def _detect_bullet_docx(spans: List[dict]) -> str:
    text_blob = " ".join(s["text"] for s in spans)
    for ch in ("•", "▪", "–", "→", "·"):
        if ch in text_blob:
            return ch
    if re.search(r"^\s*[-–—]\s+\S", text_blob, re.MULTILINE):
        return "–"
    return "•"


def _extract_dominant_font(spans: List[dict], body_size: float) -> str:
    fonts = [
        s["font"]
        for s in spans
        if abs(s["size"] - body_size) < 1.5 and s.get("font")
    ]
    if not fonts:
        fonts = [s["font"] for s in spans if s.get("font")]
    if not fonts:
        return "Calibri, Arial, Helvetica, sans-serif"
    name, _ = Counter(fonts).most_common(1)[0]
    return f"{name}, Arial, Helvetica, sans-serif"


# ─────────────────────────────────────────────────────────────────────────────
#  Small stats helpers
# ─────────────────────────────────────────────────────────────────────────────


def _median(vals: List[float]) -> float:
    if not vals:
        return 10.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return float(s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2)


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 12.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _mode(items: List[str]) -> str:
    if not items:
        return "#333333"
    return Counter(items).most_common(1)[0][0]