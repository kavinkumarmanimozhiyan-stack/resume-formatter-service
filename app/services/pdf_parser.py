"""
pdf_parser.py
─────────────────────────────────────────────────────────────────────────────
Extracts plain text + structured design tokens from a PDF.

LOGO PLACEMENT FIX (this version):
  Previously, classify_logo_image() computed real page coordinates
  (x0/y0/x1/y1) for the winning logo candidate, but extract_design_tokens()
  discarded them — only bytes_b64/ext made it into tokens["logo"]. That
  meant gemini_service had no idea WHERE on the template the logo actually
  was, so Pass 2 fell back to a generic "top of header" guess regardless of
  the logo's true position.

  Now tokens["logo"] additionally includes:
    x_pct, y_pct       — top-left corner, as % of page width/height
    width_pct, height_pct — size, as % of page width/height
    zone               — "sidebar" | "header" | "main", inferred by
                          intersecting the logo's bbox with the already-
                          detected sidebar band (and header zone) so the
                          prompt can say e.g. "inside the sidebar, near the
                          top" instead of guessing.

  This dict is consumed by gemini_service._generate_html() to build a
  placement-aware LOGO instruction instead of a generic one.
─────────────────────────────────────────────────────────────────────────────
"""

import base64
import io
from typing import Optional

import fitz   # PyMuPDF


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from every page of a PDF."""
    print(f"[PDF_PARSER] extract_text_from_pdf: {file_path}")
    doc = fitz.open(file_path)
    parts = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        parts.append(text)
        print(f"[PDF_PARSER] page {page_num}: {len(text)} chars")
    doc.close()
    result = "\n".join(parts).strip()
    print(f"[PDF_PARSER] total text: {len(result)} chars")
    return result


def rasterize_pdf_pages(file_path: str, dpi: int = 150, max_pages: int = 3) -> list[str]:
    """
    Render up to `max_pages` of the PDF as PNG images encoded as base64.
    Gemini Vision uses these as ground-truth visual anchors.
    """
    print(f"[PDF_PARSER] rasterize_pdf_pages: {file_path} @ {dpi}dpi")
    doc = fitz.open(file_path)
    images_b64: list[str] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for idx, page in enumerate(doc):
        if idx >= max_pages:
            break
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        images_b64.append(base64.b64encode(png_bytes).decode("utf-8"))
        print(f"[PDF_PARSER] rasterized page {idx}: {len(png_bytes):,} bytes")
    doc.close()
    return images_b64


def extract_images_from_pdf(file_path: str, max_images: int = 30) -> list[dict]:
    """
    Return embedded raster images (actual XObject images, not the
    rasterized page) with their placement on the page.

    Each item:
        {
            "bytes": <raw image bytes>,
            "ext": "png" | "jpeg" | ...,
            "x0", "y0", "x1", "y1": <page-space pt coords>,
            "width", "height": <pt>,
            "page_num": <0-based>,
        }
    """
    print(f"[PDF_PARSER] extract_images_from_pdf: {file_path}")
    doc = fitz.open(file_path)
    images: list[dict] = []

    for page_num, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
            except Exception as e:
                print(f"[PDF_PARSER] skip xref={xref}: extract_image failed: {e}")
                continue

            try:
                rects = page.get_image_rects(xref)
            except Exception as e:
                print(f"[PDF_PARSER] skip xref={xref}: get_image_rects failed: {e}")
                continue
            if not rects:
                continue

            rect = rects[0]
            images.append({
                "bytes": base["image"],
                "ext": base.get("ext", "png"),
                "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
                "width": rect.width, "height": rect.height,
                "page_num": page_num,
            })
            if len(images) >= max_images:
                print(f"[PDF_PARSER] hit max_images={max_images}, stopping scan")
                doc.close()
                return images

    doc.close()
    print(f"[PDF_PARSER] extracted {len(images)} embedded image(s)")
    return images


def classify_logo_image(
    images: list[dict],
    page_w: float,
    page_h: float,
    header_zone_pct: float = 0.30,
) -> Optional[dict]:
    """
    Pick the most likely "company logo" image from a set of embedded
    images extracted from a PDF template.

    Heuristic:
      1. Must be small relative to the page — under ~15% of page width
         AND height. This excludes full-bleed backgrounds and headshot
         photos, which are typically much larger.
      2. Roughly square-ish aspect ratio (logos usually are; a thin
         divider graphic or a wide banner usually isn't).
      3. Prefer images positioned in the top portion of the page
         (header_zone_pct of page height) — this is where a template's
         brand mark typically lives.
      4. If nothing qualifies in the header zone, fall back to the
         smallest qualifying image anywhere on the page (covers logos
         placed next to a job-entry area instead of the header).

    Returns None if no image looks like a plausible logo.

    NOTE: this heuristic is a best-effort filter, not a guarantee — a
    small square contact icon (e.g. next to an email/phone line) can
    still pass these checks. If you're seeing wrong logos picked (not
    just wrong placement), tightening this function is the place to do
    it — e.g. requiring the image to NOT be immediately adjacent to
    contact-info text, or cross-checking against a minimum size floor
    so 12x12px icon glyphs are excluded outright.
    """
    if not images:
        return None

    max_w = page_w * 0.15
    max_h = page_h * 0.15

    def size_ok(im: dict) -> bool:
        return im["width"] <= max_w and im["height"] <= max_h and im["width"] > 4 and im["height"] > 4

    def aspect_ok(im: dict) -> bool:
        ratio = im["width"] / im["height"] if im["height"] else 0
        return 0.4 <= ratio <= 2.5

    candidates = [im for im in images if size_ok(im) and aspect_ok(im)]
    if not candidates:
        print("[PDF_PARSER] classify_logo_image: no image passed size/aspect filter")
        return None

    header_limit = page_h * header_zone_pct
    header_candidates = [im for im in candidates if im["y0"] <= header_limit]

    pool = header_candidates if header_candidates else candidates
    # Smallest first — a smaller, tightly-cropped mark is more likely to be
    # a logo than a larger decorative graphic that happened to pass the filter.
    pool = sorted(pool, key=lambda im: (im["width"] * im["height"], im["y0"]))

    winner = pool[0]
    print(
        f"[PDF_PARSER] classify_logo_image: selected image on page={winner['page_num']} "
        f"pos=({winner['x0']:.1f},{winner['y0']:.1f}) size=({winner['width']:.1f}x{winner['height']:.1f})"
    )
    return winner


def _classify_logo_zone(
    logo_image: dict,
    page_w: float,
    page_h: float,
    sidebar: dict,
    header_zone_pct: float = 0.25,
) -> str:
    """
    Decide which layout region the logo actually sits in, by intersecting
    its bbox with the already-detected sidebar band and the header zone.
    This is what lets the Pass-2 prompt say "place it in the sidebar" /
    "place it in the header" / "place it in the main column" instead of
    always defaulting to "top of header".

    Priority: sidebar (if the logo's x-range falls inside the sidebar
    column) > header (if it's in the top header_zone_pct of the page,
    regardless of column) > main (everything else).
    """
    if sidebar.get("has_sidebar") and sidebar.get("width_px"):
        sidebar_width = sidebar["width_px"]
        if sidebar["side"] == "left":
            sidebar_x0, sidebar_x1 = 0, sidebar_width
        else:
            sidebar_x0, sidebar_x1 = page_w - sidebar_width, page_w

        # Logo counts as "in the sidebar" if the majority of its width
        # falls within the sidebar's x-range.
        overlap_x0 = max(logo_image["x0"], sidebar_x0)
        overlap_x1 = min(logo_image["x1"], sidebar_x1)
        overlap_w = max(0.0, overlap_x1 - overlap_x0)
        logo_w = max(logo_image["x1"] - logo_image["x0"], 1e-6)
        if overlap_w / logo_w > 0.5:
            return "sidebar"

    if logo_image["y0"] <= page_h * header_zone_pct:
        return "header"

    return "main"


def extract_design_tokens(file_path: str) -> dict:
    """
    Analyse the PDF and return a structured dict of design tokens.
    """
    print(f"[PDF_PARSER] extract_design_tokens: {file_path}")
    doc = fitz.open(file_path)
    page = doc[0]
    page_w = page.rect.width
    page_h = page.rect.height
    print(f"[PDF_PARSER] page size: {page_w:.1f} x {page_h:.1f} pts")

    drawings = page.get_drawings()

    sidebar = _detect_sidebar(page, page_w, page_h, drawings)
    print(f"[PDF_PARSER] sidebar: {sidebar}")

    # Collect spans from ALL pages so multi-page resumes don't lose sections
    spans: list[dict] = []
    for p in doc:
        spans.extend(_collect_spans(p))
    print(f"[PDF_PARSER] total spans (all pages): {len(spans)}")

    sizes = sorted([s["size"] for s in spans if s["size"] > 0])
    body_size    = _median(sizes) if sizes else 10.0
    name_size    = max(sizes) if sizes else 24.0
    heading_size = _percentile(sizes, 85) if sizes else 12.0
    print(f"[PDF_PARSER] sizes — body:{body_size:.1f} heading:{heading_size:.1f} name:{name_size:.1f}")

    sections = _extract_sections(spans, body_size)
    print(f"[PDF_PARSER] sections: {sections}")

    color_palette = _extract_palette(spans)
    accent_color  = _pick_accent(color_palette)
    print(f"[PDF_PARSER] accent: {accent_color}  palette: {color_palette}")

    header_info = _analyze_header(spans, page_h, page_w, name_size, drawings, sidebar)
    print(f"[PDF_PARSER] header: {header_info}")

    heading_info = _analyze_heading_style(spans, body_size, heading_size)
    print(f"[PDF_PARSER] heading style: {heading_info}")

    body_colors = [s["color_hex"] for s in spans if abs(s["size"] - body_size) < 1.5]
    body_color  = _mode(body_colors) if body_colors else "#333333"

    bullet_char = _detect_bullet(spans)
    print(f"[PDF_PARSER] bullet: {bullet_char!r}")

    # Font family extraction
    font_family = _extract_dominant_font(spans, body_size)
    print(f"[PDF_PARSER] font family: {font_family}")

    # ── Logo extraction (WITH geometry + zone, so placement can be exact) ───
    # Failure here must never break the main pipeline — a missing/failed
    # logo extraction just means the downstream placeholder tag is skipped.
    logo_token: dict = {"present": False, "bytes_b64": None, "ext": None}
    try:
        images = extract_images_from_pdf(file_path)
        logo_image = classify_logo_image(images, page_w, page_h)
        if logo_image:
            x_pct = round(logo_image["x0"] / page_w * 100, 1)
            y_pct = round(logo_image["y0"] / page_h * 100, 1)
            width_pct = round(logo_image["width"] / page_w * 100, 1)
            height_pct = round(logo_image["height"] / page_h * 100, 1)
            zone = _classify_logo_zone(logo_image, page_w, page_h, sidebar)

            logo_token = {
                "present": True,
                "bytes_b64": base64.b64encode(logo_image["bytes"]).decode("ascii"),
                "ext": logo_image["ext"],
                "x_pct": x_pct,
                "y_pct": y_pct,
                "width_pct": width_pct,
                "height_pct": height_pct,
                "zone": zone,
            }
            print(
                f"[PDF_PARSER] logo geometry: x={x_pct}% y={y_pct}% "
                f"w={width_pct}% h={height_pct}% zone={zone}"
            )
    except Exception as e:
        print(f"[PDF_PARSER] WARNING: logo extraction failed: {e}")
    print(f"[PDF_PARSER] logo: present={logo_token['present']}")

    if sidebar.get("has_sidebar"):
        layout_type = f"sidebar-{sidebar['side']}"
    else:
        layout_type = "single-column"

    doc.close()

    return {
        "layout_type":     layout_type,
        "sidebar":         sidebar,
        "header":          header_info,
        "section_heading": heading_info,
        "body": {
            "color":   body_color,
            "size_pt": body_size,
        },
        "accent_color":    accent_color,
        "color_palette":   color_palette,
        "sections":        sections,
        "section_order":   sections,
        "bullet_char":     bullet_char,
        "font_family":     font_family,
        "logo":            logo_token,
        "typography": {
            "name_size":    name_size,
            "heading_size": heading_size,
            "body_size":    body_size,
            "font_family":  font_family,
        },
        "page_size": {
            "width_pt":  page_w,
            "height_pt": page_h,
        },
    }


def extract_format_from_pdf(file_path: str) -> dict:
    """Legacy wrapper kept for backward compat."""
    tokens = extract_design_tokens(file_path)
    sidebar = tokens.get("sidebar", {})
    return {
        "page_count": 1,
        "layout": {
            "is_two_column":   sidebar.get("has_sidebar", False),
            "column_boundary": sidebar.get("width_px"),
        },
        "sidebar":       sidebar,
        "typography":    tokens["typography"],
        "color_palette": tokens["color_palette"],
        "sections":      tokens["sections"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_spans(page) -> list[dict]:
    """Return a flat list of all text spans on a page with metadata."""
    result = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                size      = span.get("size", 0)
                flags     = span.get("flags", 0)
                font      = span.get("font", "")
                color_int = span.get("color", 0)
                bbox      = span.get("bbox", [0, 0, 0, 0])

                is_bold   = bool(flags & 16) or "bold" in font.lower() or "black" in font.lower()
                is_italic = bool(flags & 2)  or "italic" in font.lower() or "oblique" in font.lower()
                color_hex = f"#{color_int:06x}"

                result.append({
                    "text":      text,
                    "size":      size,
                    "is_bold":   is_bold,
                    "is_italic": is_italic,
                    "color_int": color_int,
                    "color_hex": color_hex,
                    "font":      font,
                    "bbox":      bbox,
                    "x0": bbox[0], "y0": bbox[1],
                    "x1": bbox[2], "y1": bbox[3],
                    "block_bbox": [bx0, by0, bx1, by1],
                })
    return result


def _detect_sidebar(page, page_w: float, page_h: float, drawings: list) -> dict:
    candidates = []
    for draw in drawings:
        rect = draw.get("rect")
        fill = draw.get("fill")
        if not rect or not fill:
            continue
        w, h = rect.width, rect.height
        # Lowered from 0.5 -> 0.30: template placeholder content is often
        # short, so a genuine full-height sidebar can measure well under
        # 50% of page height. Width/shape is a far more reliable signal
        # of "this is a sidebar panel" than its incidental content height.
        if h < page_h * 0.30 or w < 20 or w > page_w * 0.45:
            continue
        r, g, b = (int(c * 255) for c in fill[:3])
        if r > 230 and g > 230 and b > 230:
            continue
        candidates.append({"rect": rect, "w": w, "h": h, "hex": f"#{r:02x}{g:02x}{b:02x}"})
    

    if candidates:
        # Pick the tallest, breaking ties by area
        best = max(candidates, key=lambda c: (c["h"], c["w"] * c["h"]))
        rect = best["rect"]
        side = "left" if rect.x0 < page_w * 0.15 else "right"
        return {
            "has_sidebar": True,
            "side":        side,
            "bg_color":    best["hex"],
            "width_px":    int(best["w"]),
            "width_pct":   round(best["w"] / page_w * 100, 1),
        }

    # Fallback: column-distribution heuristic
    page_dict = page.get_text("dict")
    mid = page_w / 2
    left_xs, right_xs = [], []
    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue
        cx = (block["bbox"][0] + block["bbox"][2]) / 2
        if cx < mid:
            left_xs.append(block["bbox"][2])
        else:
            right_xs.append(block["bbox"][0])
    if left_xs and right_xs:
        left_max  = max(left_xs)
        right_min = min(right_xs)
        gap       = right_min - left_max
        if gap > 10 and left_max < page_w * 0.42:
            return {
                "has_sidebar": True,
                "side":        "left",
                "bg_color":    None,
                "width_px":    int(left_max),
                "width_pct":   round(left_max / page_w * 100, 1),
            }

    return {"has_sidebar": False}


def _extract_sections(spans: list[dict], body_size: float) -> list[str]:
    """Return likely section heading texts in page order, dedup case-insensitive."""
    heading_threshold = body_size * 1.25
    sections: list[str] = []
    seen: set[str] = set()

    for s in spans:
        text       = s["text"].strip()
        size       = s["size"]
        upper      = text.isupper()
        word_count = len(text.split())

        if len(text) > 40:
            continue
        if any(c.isdigit() for c in text[:4]):
            continue
        if text.lower() in {
            "resume", "curriculum vitae", "cv", "page",
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


def _analyze_header(spans, page_h, page_w, name_size, drawings, sidebar) -> dict:
    """Analyse top 25% of the page to infer header styling."""
    header_zone_y = page_h * 0.25
    header_spans  = [s for s in spans if s["y0"] < header_zone_y]

    name_spans       = [s for s in header_spans if abs(s["size"] - name_size) < 2]
    name_color       = name_spans[0]["color_hex"] if name_spans else "#000000"
    name_size_actual = name_spans[0]["size"]      if name_spans else name_size

    # Header bg from drawings
    bg_color = None
    sidebar_x_range = None
    if sidebar.get("has_sidebar") and sidebar.get("width_px"):
        if sidebar["side"] == "left":
            sidebar_x_range = (0, sidebar["width_px"])
        else:
            sidebar_x_range = (page_w - sidebar["width_px"], page_w)

    for draw in drawings:
        rect = draw.get("rect")
        fill = draw.get("fill")
        if not rect or not fill:
            continue
        w = rect.width
        if w < page_w * 0.4:
            continue
        if rect.y0 > header_zone_y or rect.y1 < 5:
            continue
        # Skip if rectangle is essentially the sidebar
        if sidebar_x_range:
            sx0, sx1 = sidebar_x_range
            if abs(rect.x0 - sx0) < 10 and abs(rect.x1 - sx1) < 10:
                continue
        r, g, b = (int(c * 255) for c in fill[:3])
        if r > 230 and g > 230 and b > 230:
            continue
        bg_color = f"#{r:02x}{g:02x}{b:02x}"
        break

    # Infer from sidebar color when name text is light
    if bg_color is None and sidebar.get("has_sidebar") and sidebar.get("bg_color"):
        if name_spans:
            nc  = name_spans[0]["color_int"]
            r_n = (nc >> 16) & 0xFF
            g_n = (nc >> 8)  & 0xFF
            b_n = nc & 0xFF
            if r_n > 180 and g_n > 180 and b_n > 180:
                bg_color = sidebar["bg_color"]

    return {
        "bg_color":     bg_color,
        "name_color":   name_color,
        "name_size_pt": name_size_actual,
    }


def _analyze_heading_style(spans, body_size, heading_size) -> dict:
    threshold = body_size * 1.2
    heading_spans = [
        s for s in spans
        if (s["size"] >= threshold or s["text"].isupper())
        and len(s["text"]) < 40
        and not any(c.isdigit() for c in s["text"][:4])
        and len(s["text"].split()) <= 5
    ]
    if not heading_spans:
        return {
            "color":     "#333333",
            "bg_color":  None,
            "uppercase": False,
            "bold":      True,
            "size_pt":   heading_size,
        }
    colors = [s["color_hex"] for s in heading_spans]
    uppers = [s["text"].isupper() for s in heading_spans]
    bolds  = [s["is_bold"] for s in heading_spans]
    return {
        "color":     _mode(colors),
        "bg_color":  None,
        "uppercase": sum(uppers) > len(uppers) / 2,
        "bold":      sum(bolds)  > len(bolds)  / 2,
        "size_pt":   heading_size,
    }


def _extract_palette(spans) -> list[str]:
    freq: dict[str, int] = {}
    for s in spans:
        c = s["color_int"]
        if c < 0x222222 or c > 0xdddddd:
            continue
        h = s["color_hex"]
        freq[h] = freq.get(h, 0) + 1
    return sorted(freq, key=lambda h: -freq[h])[:8]


def _pick_accent(palette) -> str:
    if not palette:
        return "#2c3e50"
    def saturation(hx: str) -> float:
        r = int(hx[1:3], 16) / 255
        g = int(hx[3:5], 16) / 255
        b = int(hx[5:7], 16) / 255
        return max(r, g, b) - min(r, g, b)
    return max(palette, key=saturation)


def _detect_bullet(spans) -> str:
    candidates = ["•", "▪", "▸", "▶", "→", "–", "-", "◦", "○", "■", "★"]
    freq: dict[str, int] = {}
    for s in spans:
        t = s["text"].strip()
        if t in candidates:
            freq[t] = freq.get(t, 0) + 1
        elif len(t) > 1 and t[0] in candidates:
            freq[t[0]] = freq.get(t[0], 0) + 1
    if not freq:
        return "•"
    return max(freq, key=lambda k: freq[k])


def _extract_dominant_font(spans, body_size) -> str:
    """Pick the most-used font family for body-sized text."""
    freq: dict[str, int] = {}
    for s in spans:
        if abs(s["size"] - body_size) < 1.5:
            f = s["font"]
            # Normalize: strip subset prefix like "ABCDEF+FontName"
            if "+" in f:
                f = f.split("+", 1)[1]
            # Strip weight suffixes for family name
            f = f.split("-")[0].split(",")[0].strip()
            if f:
                freq[f] = freq.get(f, 0) + 1
    if not freq:
        return "Arial, Helvetica, sans-serif"
    top = max(freq, key=lambda k: freq[k])
    # Map common PDF fonts to web-safe equivalents
    mapping = {
        "Helvetica": "Helvetica, Arial, sans-serif",
        "Arial":     "Arial, Helvetica, sans-serif",
        "Times":     "'Times New Roman', Times, serif",
        "TimesNewRoman": "'Times New Roman', Times, serif",
        "Calibri":   "Calibri, 'Segoe UI', Arial, sans-serif",
        "Cambria":   "Cambria, Georgia, serif",
        "Georgia":   "Georgia, 'Times New Roman', serif",
        "Garamond":  "Garamond, Georgia, serif",
        "Verdana":   "Verdana, Geneva, sans-serif",
        "Tahoma":    "Tahoma, Geneva, sans-serif",
        "Roboto":    "Roboto, Arial, sans-serif",
        "OpenSans":  "'Open Sans', Arial, sans-serif",
        "Lato":      "Lato, Arial, sans-serif",
        "Montserrat":"Montserrat, Arial, sans-serif",
    }
    for key, val in mapping.items():
        if key.lower() in top.lower():
            return val
    return f"'{top}', Arial, Helvetica, sans-serif"


def _median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(values, pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]


def _mode(values):
    if not values:
        return "#333333"
    freq: dict[str, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    return max(freq, key=lambda k: freq[k])