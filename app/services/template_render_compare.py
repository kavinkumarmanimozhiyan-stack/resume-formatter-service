from __future__ import annotations

import io
import math
from typing import Optional

import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering (PASS 3) — SELF-CONTAINED PLACEHOLDER, see wiring note above
# ─────────────────────────────────────────────────────────────────────────────

def render_html_to_pdf_bytes(html: str, page_size: str = "A4") -> bytes:
    from . import browser_pool

    return browser_pool.render_pdf(html, {
        "print_background": True,
        "prefer_css_page_size": True,
    })


def pdf_bytes_to_png_images(pdf_bytes: bytes, dpi: int = 220, max_pages: int = 3) -> list[bytes]:
    
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed. Install it, or route this "
            "through your existing pdf_parser rasterization utility instead."
        )

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: list[bytes] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))

    return images


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry measurement (PASS 4, part 1) — pixel-scan for color blocks
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


def _decode_rgb(png_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))


def _decode_gray(png_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("L"))


def find_color_block_bbox(arr: np.ndarray, hex_color: str, tol: int = 22) -> Optional[dict]:
    """`arr` is an already-decoded RGB array (see `_decode_rgb`) — callers
    share one decode per image instead of each re-decoding the same PNG
    bytes (this used to be called up to ~4x per generated image per
    similarity score)."""
    from scipy import ndimage

    target = np.array(_hex_to_rgb(hex_color))

    diff = np.abs(arr.astype(int) - target.astype(int))
    mask = np.all(diff <= tol, axis=-1)

    if not mask.any():
        return None

    h, w = arr.shape[0], arr.shape[1]

    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    largest_size = sizes[largest_label - 1]

    if largest_size < 0.01 * h * w:
        return None

    ys, xs = np.where(labeled == largest_label)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return {
        "x_pct": round(x0 / w * 100, 2),
        "y_pct": round(y0 / h * 100, 2),
        "width_pct": round((x1 - x0) / w * 100, 2),
        "height_pct": round((y1 - y0) / h * 100, 2),
    }


def compute_geometry_diffs(tokens: dict, generated_rgb: np.ndarray, page_height_pt: float = 842.0, page_width_pt: float = 595.0) -> list[dict]:

    diffs: list[dict] = []
    blocks = tokens.get("blocks", []) or []
    sidebar = tokens.get("sidebar", {}) or {}
    header = tokens.get("header", {}) or {}

    color_lookup = {
        "sidebar": sidebar.get("bg_color"),
        "header": header.get("bg_color"),
    }

    for block in blocks:
        name = block.get("name")
        hex_color = color_lookup.get(name)
        if not hex_color:
            continue

        actual = find_color_block_bbox(generated_rgb, hex_color)
        expected = {
            "x_pct": block.get("x_pct", 0),
            "y_pct": block.get("y_pct", 0),
            "width_pct": block.get("width_pct", 0),
        }

        if actual is None:
            diffs.append({
                "element": name,
                "expected": expected,
                "actual": None,
                "delta": None,
                "issue": f"'{name}' fill color {hex_color} was not found at all in the generated render.",
            })
            continue

        def pct_to_pt(pct: float, axis: str) -> float:
            return round(pct / 100 * (page_width_pt if axis == "x" else page_height_pt), 1)

        delta = {
            "x_pt": round(pct_to_pt(actual["x_pct"], "x") - pct_to_pt(expected["x_pct"], "x"), 1),
            "y_pt": round(pct_to_pt(actual["y_pct"], "y") - pct_to_pt(expected["y_pct"], "y"), 1),
            "width_pt": round(pct_to_pt(actual["width_pct"], "x") - pct_to_pt(expected["width_pct"], "x"), 1),
        }

        # Only report as an issue if the delta exceeds a small tolerance
        # (measurement/anti-aliasing noise) — avoid manufacturing "corrections"
        # for sub-pixel-level differences that aren't visually meaningful.
        tolerance_pt = 4.0
        if max(abs(v) for v in delta.values()) > tolerance_pt:
            diffs.append({
                "element": name,
                "expected": {k: pct_to_pt(v, "x" if k == "x_pct" else ("y" if k == "y_pct" else "x")) for k, v in expected.items()},
                "actual": {
                    "x": pct_to_pt(actual["x_pct"], "x"),
                    "y": pct_to_pt(actual["y_pct"], "y"),
                    "width": pct_to_pt(actual["width_pct"], "x"),
                },
                "delta": delta,
            })

    return diffs


# ─────────────────────────────────────────────────────────────────────────────
#  Structural / color / typography scoring (PASS 4, part 2)
# ─────────────────────────────────────────────────────────────────────────────

def row_ink_profile(arr: np.ndarray, bins: int = 60) -> np.ndarray:
    """`arr` is an already-decoded grayscale array (see `_decode_gray`)."""
    h = arr.shape[0]
    band_edges = np.linspace(0, h, bins + 1).astype(int)
    profile = np.zeros(bins)
    for i in range(bins):
        band = arr[band_edges[i]:band_edges[i + 1]]
        if band.size == 0:
            continue
        profile[i] = float(np.mean(band < 245))  # fraction of non-white pixels
    return profile


def compute_structure_score(template_gray: np.ndarray, generated_gray: np.ndarray, bins: int = 60) -> float:

    p1 = row_ink_profile(template_gray, bins)
    p2 = row_ink_profile(generated_gray, bins)
    if np.std(p1) == 0 or np.std(p2) == 0:
        return 1.0 if np.allclose(p1, p2) else 0.5
    corr = float(np.corrcoef(p1, p2)[0, 1])
    return max(0.0, (corr + 1) / 2)  # map [-1,1] -> [0,1]


def compute_color_score(tokens: dict, generated_rgb: np.ndarray) -> float:
    """
    Average color-match fraction across every known fill color in the
    template tokens (sidebar, header, accent) — did the generated render
    actually use these colors anywhere, and at roughly the right place?
    """
    sidebar = tokens.get("sidebar", {}) or {}
    header = tokens.get("header", {}) or {}
    colors = [c for c in (sidebar.get("bg_color"), header.get("bg_color"), tokens.get("accent_color")) if c]
    if not colors:
        return 1.0  # nothing to check (e.g. plain single-column template)

    scores = []
    for hex_color in colors:
        bbox = find_color_block_bbox(generated_rgb, hex_color, tol=30)
        scores.append(1.0 if bbox is not None else 0.0)
    return sum(scores) / len(scores)


def compute_geometry_score(diffs: list[dict], page_width_pt: float = 595.0) -> float:
 
    if not diffs:
        return 1.0
    penalties = []
    for d in diffs:
        if d.get("delta") is None:
            penalties.append(1.0)  # element missing entirely = max penalty
            continue
        max_delta = max(abs(v) for v in d["delta"].values())
        penalties.append(min(1.0, max_delta / page_width_pt))
    return max(0.0, 1.0 - (sum(penalties) / len(penalties)))


def compute_typography_score(template_gray: np.ndarray, generated_gray: np.ndarray) -> float:

    def line_spacing_estimate(gray_arr: np.ndarray) -> float:
        profile = row_ink_profile(gray_arr, bins=200)
        peaks = [i for i in range(1, len(profile) - 1) if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > 0.05]
        if len(peaks) < 2:
            return 0.0
        gaps = np.diff(peaks)
        return float(np.mean(gaps))

    s1 = line_spacing_estimate(template_gray)
    s2 = line_spacing_estimate(generated_gray)
    if s1 == 0 or s2 == 0:
        return 0.75  # insufficient signal — neutral-ish score, not a hard fail
    ratio = min(s1, s2) / max(s1, s2)
    return ratio


def compute_similarity_score(
    template_png: bytes,
    generated_png: bytes,
    tokens: dict,
    weights: Optional[dict] = None,
) -> dict:
    # Decode each image once (RGB for color/geometry checks, grayscale for
    # structure/typography) and share the arrays across every sub-score
    # instead of each one re-decoding the same PNG bytes independently.
    weights = weights or {"geometry": 0.60, "structure": 0.05, "color": 0.25, "typography": 0.10}

    generated_rgb = _decode_rgb(generated_png)
    template_gray = _decode_gray(template_png)
    generated_gray = _decode_gray(generated_png)

    diffs = compute_geometry_diffs(tokens, generated_rgb)
    geometry_score = compute_geometry_score(diffs)
    structure_score = compute_structure_score(template_gray, generated_gray)
    color_score = compute_color_score(tokens, generated_rgb)
    typography_score = compute_typography_score(template_gray, generated_gray)

    overall = (
        geometry_score * weights["geometry"]
        + structure_score * weights["structure"]
        + color_score * weights["color"]
        + typography_score * weights["typography"]
    )

    return {
        "overall_score": round(overall, 4),
        "geometry_score": round(geometry_score, 4),
        "structure_score": round(structure_score, 4),
        "color_score": round(color_score, 4),
        "typography_score": round(typography_score, 4),
        "diffs": diffs,
        "num_issues": len(diffs),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Correction instructions (PASS 5)
# ─────────────────────────────────────────────────────────────────────────────

def build_corrections_text(diffs: list[dict]) -> str:
    
    if not diffs:
        return ""

    lines = ["CORRECTIONS:"]
    for i, d in enumerate(diffs, start=1):
        element = d["element"]
        if d.get("actual") is None:
            lines.append(
                f"{i}. The '{element}' fill color/region could not be found at all in "
                f"the generated output. Verify it is rendered with the exact measured "
                f"color and position — it may be missing, mis-colored, or positioned "
                f"far outside its expected bounds."
            )
            continue

        delta = d["delta"]
        parts = []
        if abs(delta.get("width_pt", 0)) > 4:
            direction = "wider" if delta["width_pt"] > 0 else "narrower"
            parts.append(
                f"'{element}' is {abs(delta['width_pt']):.1f}pt {direction} than the "
                f"template. {'Reduce' if direction == 'wider' else 'Increase'} its width "
                f"by {abs(delta['width_pt']):.1f}pt."
            )
        if abs(delta.get("x_pt", 0)) > 4:
            direction = "too far right" if delta["x_pt"] > 0 else "too far left"
            parts.append(
                f"'{element}' starts {abs(delta['x_pt']):.1f}pt {direction}. "
                f"Adjust its horizontal offset by {-delta['x_pt']:.1f}pt."
            )
        if abs(delta.get("y_pt", 0)) > 4:
            direction = "too low" if delta["y_pt"] > 0 else "too high"
            parts.append(
                f"'{element}' starts {abs(delta['y_pt']):.1f}pt {direction}. "
                f"Adjust its vertical offset by {-delta['y_pt']:.1f}pt."
            )
        if parts:
            lines.append(f"{i}. " + " ".join(parts))

    lines.append("")
    lines.append(
        "Apply ONLY these corrections. Do not change colors, section content, "
        "typography, or any property not listed above."
    )
    return "\n".join(lines)
