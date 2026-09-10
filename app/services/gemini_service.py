"""
gemini_service.py
─────────────────────────────────────────────────────────────────────────────
Two-pass Gemini pipeline for pixel-perfect resume formatting.

Pass 1 — _extract_tokens_via_vision()
    • Sends rasterized page images of the TEMPLATE (high DPI)
    • Returns a JSON design-token dict that includes GEOMETRIC bounding
      boxes for every major visual block, not just qualitative descriptions.
    • The template's textual content (placeholder / sample text) is
      explicitly ignored — Gemini is told this is a design template only.

Pass 2 — _generate_html()
    • Receives merged tokens + rendered TEMPLATE page images + Resume Y text
    • Converts the bounding-box tokens into an explicit HTML <table>
      column scaffold (exact %/pt values), instead of relying on Gemini
      to eyeball proportions or use Grid/Flexbox (unreliable across
      HTML-to-PDF renderers)
    • Embeds REQUIRED_CSS verbatim
    • Generates pixel-accurate HTML using ONLY Resume Y's content
    • Leaves a LOGO_PLACEHOLDER marker where the template's logo goes,
      rather than attempting to redraw it (see LOGO SPLICING below)

Pass 3 (optional, stubbed) — _verify_and_repair()
    • Renders the Pass-2 HTML to an image, shows [template image, rendered
      image] to Gemini side by side, asks for a structured list of alignment
      deltas, and does one corrective regeneration pass if deltas are
      significant. This is the single biggest lever for true pixel fidelity,
      but requires hooking into your existing HTML→image/PDF renderer.
─────────────────────────────────────────────────────────────────────────────

WHY THE OLD PIPELINE UNDER-PERFORMED (~50% accuracy, alignment drift):

  1. GEMINI_MODEL was pinned to a "flash-lite" tier model. Lite/fast models
     are noticeably worse at fine-grained spatial reasoning from an image —
     they approximate proportions rather than reading them. This is very
     likely the single biggest source of error. Pass 2 (HTML generation)
     in particular should use a full-capability model.

  2. The old token schema was qualitative only (sidebar "width_estimate",
     "name_size_pt", etc.) with no actual geometry. Gemini in Pass 2 had to
     re-derive layout from prose ("sidebar + main content") instead of
     numbers, which is exactly what produces alignment drift.

  3. There was no verification step — one-shot generation, validated only
     for "is this valid HTML", never checked against the template visually.

This version addresses (1) and (2) directly, and stubs out (3).
─────────────────────────────────────────────────────────────────────────────

LOGO PLACEMENT FIX (this version):

  Previously tokens["logo"] only carried {present, bytes_b64, ext}. The
  parsers (pdf_parser.py / docx_parser.py) actually computed real position
  data for the winning logo candidate — page-relative x/y/width/height % for
  PDFs, paragraph index + table-column context for DOCX — but none of it
  reached this file, so the Pass-2 prompt always fell back to a generic
  "top of header or sidebar" instruction regardless of where the logo
  actually was in the template.

  Both parsers now also return a "zone" ("sidebar" | "header" | "main") plus
  the raw position fields. _build_logo_instruction() below reads those and
  builds a placement instruction anchored to the ACTUAL measured/observed
  location instead of a generic guess.

  Also: _splice_logo_into_html() previously logged nothing when a logo was
  extracted (tokens.logo.present == True) but Gemini never emitted the
  placeholder marker in its HTML — that failure was completely silent. It
  now logs a WARNING in that case so a "logo just isn't showing up" bug
  shows up in the logs instead of requiring a code read to discover.
─────────────────────────────────────────────────────────────────────────────

ARCHITECTURE NOTE (template-only design source):

    TEMPLATE  = visual/design source ONLY. May contain placeholder text
                like [NAME], [JOB TITLE]. That text is NEVER extracted,
                NEVER sent to Gemini as factual content, and NEVER allowed
                into the final generated resume.

    RESUME Y  = the ONLY source of actual resume content.

For backward compatibility, the public function keeps its original
parameter names (`resume_x_pdf_path`, `resume_x_text`, `resume_x_docx_path`).
Internally these are aliased to `template_*` names. `template_text` is kept
ONLY for debug logging — it is never included in any prompt sent to Gemini.
─────────────────────────────────────────────────────────────────────────────

LOGO SPLICING:

    docx_parser.extract_design_tokens() / pdf_parser.extract_design_tokens()
    now each return a `tokens["logo"]` dict:
        {
          "present": bool, "bytes_b64": str | None, "ext": str | None,
          "zone": "sidebar" | "header" | "main" | None,
          # PDF only:
          "x_pct": float | None, "y_pct": float | None,
          "width_pct": float | None, "height_pct": float | None,
          # DOCX only:
          "para_index": int | None, "total_paragraphs": int | None,
          "near_text": str | None,
        }

    This is the RAW, EXACT image bytes of the template's detected company
    logo. Gemini is NEVER shown these bytes and NEVER asked to redraw or
    describe the logo — an LLM cannot reproduce exact pixels, and asking it
    to "recreate the logo" would produce a fabricated image, not the real
    one. Instead:
      1. The Pass-2 prompt tells Gemini, if tokens.logo.present is true, to
         leave a literal marker string (LOGO_PLACEHOLDER_TOKEN) inside an
         <img> tag's src attribute at the appropriate spot — now informed
         by the real zone/position data described above.
      2. After Gemini returns HTML, _splice_logo_into_html() does a plain
         string replacement of that marker with a base64 data: URI built
         directly from the extracted bytes — guaranteeing pixel-identical
         output regardless of what the model actually generated around it.
      3. If tokens.logo.present is false, any stray placeholder the model
         inserted anyway is stripped out, so there's never a broken image.
"""

import base64
import json
import re
from typing import Optional

from google import genai
from google.genai import types

from app.config import settings
from app.services.pdf_parser import extract_design_tokens, rasterize_pdf_pages
from app.services.llm_client import (
    generate_llm_text,
    get_llm_runtime as get_text_llm_runtime,
)
from app.services.template_render_compare import (
    render_html_to_pdf_bytes,
    pdf_bytes_to_png_images,
    compute_similarity_score,
    build_corrections_text,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Iterative template-matching config (PASS 3-6)
# ─────────────────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 2          # hard cap — never loops forever
SIMILARITY_THRESHOLD = 0.92  # stop early once we're at least this close

# Vision-pass DPI: what Gemini actually sees for token/geometry extraction.
# Accuracy-critical — do not lower.
VISION_DPI = 220
# Compare-loop DPI: used only for the internal pixel-diff similarity score
# (never sent to Gemini, never shown to the user). Rasterized fresh on both
# sides at this resolution — lower than VISION_DPI on purpose, since scoring
# doesn't need print-quality detail. Keeps compare-loop memory down without
# touching what Gemini sees.
COMPARE_DPI = 150

# Marker Gemini is instructed to emit verbatim as an <img src="..."> value.
# Deliberately unusual/unique so it can never collide with real content.
LOGO_PLACEHOLDER_TOKEN = "LOGO_PLACEHOLDER_TOKEN_DO_NOT_MODIFY"


# ─────────────────────────────────────────────────────────────────────────────
#  Client + models
# ─────────────────────────────────────────────────────────────────────────────

client = None


def _get_shared_client():
    global client
    if client is None and settings.GOOGLE_API_KEY:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return client


# IMPORTANT: use a full-capability model, not a "lite"/"flash-lite" tier,
# for any pass that reasons about spatial layout from an image. Lite models
# are tuned for speed/cost and consistently under-perform on fine-grained
# visual geometry — this is very likely your #1 source of alignment error.
#
# Recommended: a current "pro" or full "flash" (non-lite) snapshot for BOTH
# passes. Swap these two constants to match whatever your account currently
# has access to; keep them pinned (not "latest") so behavior doesn't drift
# under you.
GEMINI_MODEL_VISION = "gemini-3.1-pro-preview"   # Pass 1: token/geometry extraction
GEMINI_MODEL_HTML = "gemini-3.5-flash"    # Pass 2: HTML generation


def _normalize_gemini_provider(provider: str | None) -> str:
    provider = (provider or "gemini").strip().lower()
    if provider in ("google", "googleai", "genai"):
        return "gemini"
    return provider


def _selected_text_provider(llm_settings: Optional[dict] = None) -> str:
    provider = ((llm_settings or {}).get("provider") or "gemini").strip().lower()
    if provider in ("google", "googleai", "genai"):
        return "gemini"
    if provider in ("anthropic", "claude"):
        return "claude"
    if provider in ("local", "ollama"):
        return "ollama"
    return provider


def _get_runtime(llm_settings: Optional[dict] = None, default_model: str = GEMINI_MODEL_HTML):
    """
    Resolve the Gemini client/model for this request.
    Other providers need provider-specific prompt adapters because this service
    uses Gemini multimodal parts for layout extraction.
    """
    provider = _normalize_gemini_provider((llm_settings or {}).get("provider"))
    if provider not in ("", "gemini"):
        raise ValueError(
            f"Provider '{provider}' is not supported by the visual resume pipeline yet. "
            "Use Gemini for template layout matching."
        )

    api_key = ((llm_settings or {}).get("api_key") or "").strip()
    model = ((llm_settings or {}).get("model") or "").strip() or default_model

    if api_key:
        return genai.Client(api_key=api_key), model

    shared_client = _get_shared_client()
    if not shared_client:
        raise ValueError(
            "No Google API key was provided. Supply llm_api_key in the request or set GOOGLE_API_KEY in the environment."
        )

    return shared_client, model


# ─────────────────────────────────────────────────────────────────────────────
#  CSS that MUST appear verbatim in every generated resume
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_CSS = """
/* === REQUIRED BASE — do not remove or alter === */
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
    background: #ffffff;
}

body {
    font-size: 10pt;
    color: #333333;
    line-height: 1.4;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

@page {
    margin: 0;
    size: A4 portrait;
}

@media print {
    body { margin: 0; }
    .page-wrapper { box-shadow: none !important; }
}

table.resume-layout {
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
}
table.resume-layout > tbody > tr > td {
    vertical-align: top;
}

table.resume-layout > tbody > tr > td {
    vertical-align: top;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

.section      { page-break-inside: avoid; }
.job-entry    { page-break-inside: avoid; }
.edu-item     { page-break-inside: avoid; }
.cert-item    { page-break-inside: avoid; }
.project-item { page-break-inside: avoid; }
.skill-group  { page-break-inside: avoid; }

/* NOTE: page-break-inside:avoid only applies to CONTENT rows (job/edu/etc,
   handled above via their own classes). It must NOT apply to the outer
   resume-layout row itself — that row spans the whole document and is
   frequently taller than one page by design, so forcing "avoid" on it is
   unsatisfiable and causes Chromium to drop/duplicate cell backgrounds at
   the forced break, which is what produces empty gaps in a long sidebar. */
tr { page-break-after: auto; }
table.resume-layout > tbody > tr { page-break-inside: auto; }

h1, h2, h3, h4 { page-break-after: avoid; }
p, .bullet { orphans: 2; widows: 2; page-break-inside: avoid; }
/* === END REQUIRED BASE === */
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_formatted_resume(
    resume_x_pdf_path: Optional[str],
    resume_x_text: str,
    resume_y_text: str,
    resume_x_docx_path: Optional[str] = None,
    llm_settings: Optional[dict] = None,
    max_iterations: int = MAX_ITERATIONS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> str:
    """
    Generate a formatted HTML resume:
      - Visual layout/style sourced from the TEMPLATE (formerly "Resume X")
      - Content sourced ENTIRELY from Resume Y

    Parameter names kept as `resume_x_*` for backward compatibility.
    `max_iterations`/`similarity_threshold` control the render-compare-
    correct refinement loop (PASS 3-6) — safe to leave at defaults; existing
    call sites that don't pass them are unaffected.
    """
    template_pdf_path = resume_x_pdf_path
    template_docx_path = resume_x_docx_path
    template_text = resume_x_text  # debug/logging only — NEVER sent to Gemini as content

    print(f"[GEMINI] generate_formatted_resume called")
    print(f"[GEMINI] template_pdf_path: {template_pdf_path}")
    print(f"[GEMINI] template_docx_path: {template_docx_path}")
    print(f"[GEMINI] template_text len (unused as content): {len(template_text)}")
    print(f"[GEMINI] resume_y_text len: {len(resume_y_text)}")

    # ── A. Structure tokens (PDF via PyMuPDF, or DOCX via python-docx) ───────
    pymupdf_tokens: dict = {}
    if template_docx_path:
        try:
            from app.services import docx_parser

            pymupdf_tokens = docx_parser.extract_design_tokens(template_docx_path)
            print(f"[GEMINI] DOCX parser tokens: {json.dumps(_tokens_for_log(pymupdf_tokens), indent=2)}")
        except Exception as e:
            print(f"[GEMINI] WARNING: DOCX token extraction failed: {e}")
    elif template_pdf_path:
        try:
            pymupdf_tokens = extract_design_tokens(template_pdf_path)
            print(f"[GEMINI] PyMuPDF tokens: {json.dumps(_tokens_for_log(pymupdf_tokens), indent=2)}")
        except Exception as e:
            print(f"[GEMINI] WARNING: PyMuPDF token extraction failed: {e}")

    # ── B. Load PDF bytes and rasterize pages at HIGH DPI ────────────────────
    # 150dpi is too low to reliably read small text alignment cues (kerning,
    # indent levels, sub-pixel column boundaries). Bumped to 220dpi. This
    # increases token cost per call but materially improves what the vision
    # pass can actually see.
    pdf_b64: Optional[str] = None
    page_images_b64: list[str] = []
    if template_pdf_path:
        try:
            with open(template_pdf_path, "rb") as f:
                pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
            print(f"[GEMINI] Template PDF loaded ({len(pdf_b64)} b64 chars)")
        except Exception as e:
            print(f"[GEMINI] WARNING: could not load template PDF: {e}")

        try:
            page_images_b64 = rasterize_pdf_pages(template_pdf_path, dpi=VISION_DPI, max_pages=3)
            print(f"[GEMINI] rasterized {len(page_images_b64)} template page images @{VISION_DPI}dpi")
        except Exception as e:
            print(f"[GEMINI] WARNING: rasterization failed: {e}")

    # ── C. Pass 1 — Vision token extraction (geometry + design tokens) ──────
    vision_tokens: dict = {}
    if pdf_b64 or page_images_b64:
        try:
            vision_tokens = _extract_tokens_via_vision(
                pdf_b64, page_images_b64, pymupdf_tokens, llm_settings=llm_settings
            )
            print(f"[GEMINI] Vision tokens: {json.dumps(vision_tokens, indent=2)}")
        except Exception as e:
            print(f"[GEMINI] WARNING: vision token extraction failed: {e}")

    # ── D. Merge tokens — PyMuPDF wins for colors/typography (exact bytes) ──
    merged_tokens = {**vision_tokens, **pymupdf_tokens}

    # Layout/structural + geometric keys: vision wins (PyMuPDF can't see these)
    for key in (
        "layout_type",
        "sidebar",
        "decorative_elements",
        "contact_style",
        "blocks",
        "spacing",
        "page_margin_pt",
    ):
        if key in vision_tokens and vision_tokens[key]:
            merged_tokens[key] = vision_tokens[key]

    # Section order: prefer non-empty
    if not merged_tokens.get("section_order"):
        merged_tokens["section_order"] = (
            merged_tokens.get("sections")
            or vision_tokens.get("section_order")
            or []
        )

    # ── D'. Header color: PyMuPDF is the color authority, unconditionally ────
    # This module's own design says "PyMuPDF wins on colors/typography/fonts
    # (exact bytes); vision wins on layout/structure only." The dict-spread
    # merge above only enforces that for "header" if pymupdf_tokens happens
    # to include a top-level "header" key with the field populated — if
    # PyMuPDF's extraction is partial (e.g. returns a header dict but the
    # bg_color field is simply absent rather than explicitly None), a
    # vision-hallucinated color can silently survive. Confirmed in
    # practice: a plain single-column template with NO header color at all
    # came back from the pipeline with an invented dark (#212121) header
    # band. Force PyMuPDF's bg_color reading to win whenever PyMuPDF
    # produced a header dict at all — including when its answer is "no
    # color" (None) — since "no color" is a real, common, correct answer
    # that vision should not be allowed to override.
    if isinstance(pymupdf_tokens.get("header"), dict):
        merged_tokens.setdefault("header", {})
        merged_tokens["header"]["bg_color"] = pymupdf_tokens["header"].get("bg_color")
        print(
            f"[GEMINI] Header bg_color forced to PyMuPDF's reading: "
            f"{pymupdf_tokens['header'].get('bg_color')!r} "
            f"(vision had suggested: {vision_tokens.get('header', {}).get('bg_color')!r})"
        )

    # ── D''. Logo: PyMuPDF/docx_parser is the ONLY authority — vision never
    # touches this key at all, since it isn't even in the vision JSON schema.
    # Just make sure the key always exists downstream even if extraction
    # failed or produced nothing, so _generate_html can rely on it.
    if not isinstance(merged_tokens.get("logo"), dict):
        merged_tokens["logo"] = {"present": False, "bytes_b64": None, "ext": None}

    _backfill_header_bg(merged_tokens)

    print(f"[GEMINI] Merged tokens (logo omitted from log): {json.dumps(_tokens_for_log(merged_tokens), indent=2)}")

    # ── E. Pass 2 — HTML generation (content comes ONLY from Resume Y) ───────
    html_content = _generate_html(
        tokens=merged_tokens,
        resume_y_text=resume_y_text,
        pdf_b64=pdf_b64,
        page_images_b64=page_images_b64,
        llm_settings=llm_settings,
    )

    # ── E'. Splice the real logo bytes into the HTML ─────────────────────────
    # This happens on every generation, including every refinement iteration
    # below, since each iteration calls _generate_html() again and gets a
    # fresh copy of the placeholder marker.
    html_content = _splice_logo_into_html(html_content, merged_tokens.get("logo") or {})

    # ── F. PASS 3-6 — render, compare, correct, regenerate (bounded loop) ───
    # This step is an enhancement on top of a result that's already valid
    # (Pass 2's output already passed _validate_html). If rendering or
    # comparison can't run for any reason, we log it and return the Pass-2
    # HTML as-is rather than fail the whole request — per the "fail
    # gracefully, don't silently return low-quality" requirement, we log
    # loudly instead of raising, since the fallback here is still a
    # correct, previously-working result.
    if not page_images_b64:
        print("[GEMINI] Skipping render-compare loop: no template page image available to compare against.")
        return html_content

    try:
        # Rasterize a dedicated compare-resolution copy — deliberately NOT
        # reusing page_images_b64 (that stays at VISION_DPI for Gemini).
        # Both sides of the similarity comparison must be rasterized at the
        # same DPI for compute_similarity_score's pixel-diff/color-block
        # logic to stay valid.
        compare_template_images = rasterize_pdf_pages(template_pdf_path, dpi=COMPARE_DPI, max_pages=1)
        template_png = base64.b64decode(compare_template_images[0])
    except Exception as e:
        print(f"[GEMINI] WARNING: could not rasterize/decode template page image for comparison: {e}")
        return html_content

    for iteration in range(1, max_iterations + 1):
        import time
        t0 = time.time()
        try:
            generated_pdf_bytes = render_html_to_pdf_bytes(html_content)
            generated_pages = pdf_bytes_to_png_images(generated_pdf_bytes, dpi=COMPARE_DPI, max_pages=3)
        except Exception as e:
            print(f"[GEMINI] Iteration {iteration}: render failed ({e}). Keeping last valid HTML and stopping refinement.")
            break

        if not generated_pages:
            print(f"[GEMINI] Iteration {iteration}: renderer produced 0 pages. Stopping refinement.")
            break

        try:
            score = compute_similarity_score(template_png, generated_pages[0], merged_tokens)
        except Exception as e:
            print(f"[GEMINI] Iteration {iteration}: comparison failed ({e}). Keeping last valid HTML and stopping refinement.")
            break

        render_duration = round(time.time() - t0, 2)
        print(
            f"[GEMINI] Iteration {iteration}: similarity={score['overall_score']*100:.1f}% "
            f"(geometry={score['geometry_score']*100:.1f}%, structure={score['structure_score']*100:.1f}% [diagnostic-only, low weight], "
            f"color={score['color_score']*100:.1f}%, typography={score['typography_score']*100:.1f}%) "
            f"issues={score['num_issues']} render_time={render_duration}s pages={len(generated_pages)}"
        )
        for d in score["diffs"]:
            print(f"[GEMINI]   diff: {d}")

        # Done with this iteration's raw render/rasterize buffers — release
        # before the next iteration (if any) allocates fresh ones.
        del generated_pdf_bytes, generated_pages

        if score["overall_score"] >= similarity_threshold or score["num_issues"] == 0:
            print(f"[GEMINI] Iteration {iteration}: similarity threshold reached ({similarity_threshold*100:.0f}%). Stopping.")
            break

        if iteration == max_iterations:
            print(f"[GEMINI] Reached max_iterations={max_iterations} without hitting threshold. Returning best available HTML.")
            break

        corrections_text = build_corrections_text(score["diffs"])
        if not corrections_text:
            print(f"[GEMINI] Iteration {iteration}: no actionable corrections could be derived from diffs. Stopping.")
            break

        print(f"[GEMINI] Iteration {iteration}: regenerating with {score['num_issues']} measured correction(s)...")
        try:
            html_content = _generate_html(
                tokens=merged_tokens,
                resume_y_text=resume_y_text,
                pdf_b64=pdf_b64,
                page_images_b64=page_images_b64,
                llm_settings=llm_settings,
                corrections=corrections_text,
            )
            html_content = _splice_logo_into_html(html_content, merged_tokens.get("logo") or {})
        except Exception as e:
            print(f"[GEMINI] Iteration {iteration}: regeneration failed ({e}). Keeping previous HTML and stopping.")
            break

    return html_content


# ─────────────────────────────────────────────────────────────────────────────
#  Token post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokens_for_log(tokens: dict) -> dict:
    """
    Return a copy of tokens safe to json.dumps into logs — strips the
    (potentially large) base64 logo payload so logs don't get flooded with
    an image blob.
    """
    if not isinstance(tokens, dict):
        return tokens
    copy = dict(tokens)
    logo = copy.get("logo")
    if isinstance(logo, dict) and logo.get("bytes_b64"):
        copy["logo"] = {**logo, "bytes_b64": f"<{len(logo['bytes_b64'])} b64 chars omitted>"}
    return copy


def _splice_logo_into_html(html_content: str, logo: dict) -> str:
    """
    Replace the LOGO_PLACEHOLDER_TOKEN marker (if Gemini emitted it) with a
    real base64 data: URI built from the EXACT bytes extracted from the
    template. If no logo was detected, or Gemini didn't emit the marker,
    this is a safe no-op; if Gemini emitted the marker but no logo exists
    (shouldn't happen per the prompt, but defensively handled), the stray
    tag is stripped so no broken image ever reaches the output.

    FIX: previously, if logo.present was True but Gemini simply never
    emitted the placeholder tag at all, this function returned silently
    with no log line — "logo extracted but missing from output" was
    invisible in the logs. Now that case is logged explicitly.
    """
    has_marker = LOGO_PLACEHOLDER_TOKEN in html_content

    if logo.get("present") and not has_marker:
        print(
            "[GEMINI] WARNING: logo was extracted (present=True) but the model "
            "never emitted the LOGO_PLACEHOLDER_TOKEN marker in its HTML — the "
            "logo will be MISSING from the final output. This is a generation-time "
            "omission, not a splicing failure; consider re-running or tightening "
            "the LOGO instruction in _build_logo_instruction()."
        )

    if not has_marker:
        return html_content

    if logo.get("present") and logo.get("bytes_b64"):
        ext = (logo.get("ext") or "png").lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        data_uri = f"data:{mime};base64,{logo['bytes_b64']}"
        html_content = html_content.replace(LOGO_PLACEHOLDER_TOKEN, data_uri)
        print(f"[GEMINI] Logo spliced into HTML from extracted template bytes (zone={logo.get('zone')})")
    else:
        # Model inserted the marker even though no logo was actually
        # extracted — strip the whole <img> tag so nothing broken renders.
        html_content = re.sub(
            r'<img[^>]*' + re.escape(LOGO_PLACEHOLDER_TOKEN) + r'[^>]*/?>',
            "",
            html_content,
        )
        print("[GEMINI] WARNING: model emitted logo placeholder with no logo extracted — stripped")

    return html_content


def _backfill_header_bg(tokens: dict) -> None:
    header = tokens.get("header", {}) or {}
    sidebar = tokens.get("sidebar", {}) or {}
    if header.get("bg_color"):
        return
    if not sidebar.get("has_sidebar") or not sidebar.get("bg_color"):
        return
    name_color = header.get("name_color", "#000000")
    try:
        nc = int(name_color.lstrip("#"), 16)
        r_n = (nc >> 16) & 0xFF
        g_n = (nc >> 8) & 0xFF
        b_n = nc & 0xFF
        if r_n > 180 and g_n > 180 and b_n > 180:
            tokens.setdefault("header", {})
            tokens["header"]["bg_color"] = sidebar["bg_color"]
            print(f"[GEMINI] backfilled header.bg_color = {sidebar['bg_color']}")
    except (ValueError, TypeError):
        pass


def _detect_columns(blocks: list[dict]) -> tuple[Optional[dict], list[dict]]:
    """
    Generalized geometry reader. Works for ANY uploaded template, not just a
    fixed sidebar+main pair — N side-by-side columns, no sidebar at all, or
    a full-width header row above the columns.

    Returns (header_block_or_None, column_blocks_left_to_right).

    IMPORTANT: column detection does NOT use measured block height as a
    filter. Verified against real templates: a sidebar's colored background
    in the template image is only as tall as whatever placeholder content
    happened to fill it (measured 43.8% on one template, 50.2% on another,
    on two templates that are both clearly meant to be full-height sidebar
    designs). Filtering on height would incorrectly reject genuine columns
    and — more importantly — height_pct must never be used later to cap a
    column's fill color, since real resume content is a different length
    than the template's placeholder text. Columns are identified purely by
    geometry that IS content-length-independent: they sit side by side
    (non-overlapping x-ranges) and start at the same y position.

    A block counts as the "header" if it spans nearly the full page width
    (>= ~85%) and starts at/near y=0 — a banner that columns begin below.
    """
    if not blocks:
        return None, []

    header = None
    for b in blocks:
        if b.get("y_pct", 0) <= 3 and b.get("width_pct", 0) >= 85:
            header = b
            break

    header_bottom = (header.get("y_pct", 0) + header.get("height_pct", 0)) if header else 0

    # Candidates: everything else that starts roughly where the header ends
    # (or at the top, if there's no header). No height filter.
    candidates = [
        b
        for b in blocks
        if b is not header
        and b.get("width_pct", 0) > 0
        and abs(b.get("y_pct", 0) - header_bottom) <= 5
    ]
    candidates = sorted(candidates, key=lambda b: b.get("x_pct", 0))

    # Keep only candidates whose x-ranges don't overlap — that's what makes
    # them "side by side columns" rather than stacked/overlapping regions.
    columns: list[dict] = []
    cursor = 0.0
    for b in candidates:
        x0 = b.get("x_pct", 0)
        if x0 + 1 >= cursor:  # small tolerance for measurement noise
            columns.append(b)
            cursor = x0 + b.get("width_pct", 0)

    return header, columns


def _table_widths_pct(columns: list[dict]) -> list[float]:
    """Normalize measured column widths so they sum to 100%."""
    raw = [max(b.get("width_pct", 0), 1) for b in columns]
    total = sum(raw) or 1
    return [round(w / total * 100, 1) for w in raw]


def _clamp_padding(padding_pt: dict, max_pt: float = 28.0, min_pt: float = 8.0, label: str = "") -> dict:
    """
    Defensive backstop on measured padding values. Vision-pass padding
    measurements have shown real overestimation in practice — a measured
    36pt top padding on a sidebar produced a visibly empty gap above the
    name that isn't actually in the template. Rather than trust raw vision
    output unconditionally for a value this visually sensitive, clamp to a
    reasonable range and log when clamping actually changes something, so
    it's visible in the logs rather than a silent guess.
    """
    clamped = dict(padding_pt or {})
    for side in ("top", "right", "bottom", "left"):
        val = clamped.get(side)
        if val is None:
            continue
        new_val = max(min_pt, min(max_pt, val))
        if new_val != val:
            print(f"[GEMINI] Clamped {label} padding.{side}: {val}pt -> {new_val}pt (measured value looked implausible)")
            clamped[side] = new_val
    return clamped


def _build_logo_instruction(logo: dict) -> str:
    """
    Build a placement-aware LOGO instruction for the Pass-2 prompt, using
    whatever real position data the parser managed to extract, instead of
    always defaulting to a generic "top of header" guess.

    logo may contain (depending on source format):
      PDF:  x_pct, y_pct, width_pct, height_pct, zone
      DOCX: para_index, total_paragraphs, near_text, zone
      Both: present, bytes_b64, ext, zone
    """
    if not logo.get("present"):
        return """
LOGO — no logo mark was detected in the template. Do NOT insert any logo,
icon-as-logo, or placeholder image anywhere in the output.
"""

    zone = logo.get("zone") or "header"

    if "x_pct" in logo and logo.get("x_pct") is not None:
        # PDF source — precise geometry available.
        placement_detail = (
            f"The logo was measured at approximately {logo['x_pct']}% from the "
            f"left edge of the page and {logo['y_pct']}% from the top, sized "
            f"about {logo.get('width_pct', '?')}% wide by {logo.get('height_pct', '?')}% "
            f"tall relative to the page. This places it inside the '{zone}' region "
            f"of the layout (as opposed to some other region) — treat that as the "
            f"authoritative position, not a rough guess."
        )
    elif "para_index" in logo and logo.get("para_index") is not None:
        # DOCX source — ordinal position + nearby text only.
        near = logo.get("near_text") or "(no nearby text captured)"
        placement_detail = (
            f"The logo was found at paragraph {logo['para_index']} of "
            f"{logo.get('total_paragraphs', '?')} in the source document, next to "
            f"the text \"{near}\". This places it inside the '{zone}' region of the "
            f"layout — treat that as the authoritative position, not a rough guess."
        )
    else:
        placement_detail = (
            f"Exact coordinates were not available, but the logo was classified "
            f"as belonging to the '{zone}' region of the layout — place it there, "
            f"not in some other region."
        )

    if zone == "sidebar":
        zone_rule = (
            "Place the logo INSIDE the sidebar column, at the top of the sidebar's "
            "content (above the sidebar's other content, unless the position details "
            "above clearly indicate otherwise). Do NOT place it in the full-width "
            "header band or in the main column."
        )
    elif zone == "main":
        zone_rule = (
            "Place the logo INSIDE the main (non-sidebar) column, at the relative "
            "position implied by the position details above — this is NOT a header "
            "logo, do not move it into a full-width header band."
        )
    else:  # header
        zone_rule = (
            "Place the logo INSIDE the full-width header block, at the relative "
            "horizontal position implied by the position details above (e.g. do not "
            "assume centered or left-aligned unless the details suggest that)."
        )

    return f"""
LOGO — the template contains a company/brand logo mark. {placement_detail}

{zone_rule}

You must NOT attempt to draw, describe, or recreate the logo yourself in any
way (no SVG icon, no text substitute, no "generic logo" placeholder graphic)
— the real image will be spliced in programmatically after you respond,
using the exact original file bytes. Your only job is to emit, at the
position described above, exactly this tag with NO modification to the src
value:

  <img class="tpl-logo" src="{LOGO_PLACEHOLDER_TOKEN}" alt="Logo" style="max-width:100%; height:auto; display:block;" />

Do not wrap it in extra decoration that would fight with the real logo's
own colors (no colored background box behind it unless the template itself
clearly showed one). Size it to roughly match the proportions the logo
occupied in the template.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 1 — Vision token extraction (TEMPLATE — design + GEOMETRY only)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tokens_via_vision(
    pdf_b64: Optional[str],
    page_images_b64: list[str],
    pymupdf_hints: dict,
    llm_settings: Optional[dict] = None,
) -> dict:
    """
    Ask Gemini to visually analyze the TEMPLATE and return a JSON token dict
    that includes explicit GEOMETRY (bounding boxes as % of page) for every
    major visual block, not just qualitative style descriptions. Geometry is
    what Pass 2 uses to build a <table>-based column scaffold, which is
    what actually fixes alignment drift (rendered consistently across
    HTML-to-PDF engines, unlike Grid/Flexbox).

    NOTE: logo bytes/position are intentionally NOT part of this schema and
    never sent to the vision model — logo handling is bytes-in/bytes-out via
    docx_parser/pdf_parser + _splice_logo_into_html only.
    """
    # Never leak the (potentially large) logo b64 payload into this hint
    # dump — it's irrelevant to vision's job and just wastes tokens.
    hints_for_prompt = _tokens_for_log(pymupdf_hints)
    hints_str = json.dumps(hints_for_prompt, indent=2) if hints_for_prompt else "{}"

    prompt = f"""
You are analyzing a resume template visually, as if you were measuring it
with a ruler against a percentage grid overlaid on the page (0% = top/left
edge, 100% = bottom/right edge of the page).

The attached image(s) are a DESIGN TEMPLATE ONLY — not a content source.
Ignore all placeholder/sample text (e.g. [NAME], [JOB TITLE], generic
sample names/companies/dates). Never extract factual resume content.

Your job is to output PRECISE MEASUREMENTS, not vibes. For every major
visual block, estimate its bounding box as percentages of the full page
(x_pct, y_pct = top-left corner; width_pct, height_pct = size), based on
page 1. Be as numerically precise as you can from the image — this is used
to build an actual layout scaffold, so rough guesses like "about half" are
not good enough; look carefully at where lines and edges actually fall.

CRITICAL — do not invent decorative colors that aren't in the image. A
plain resume with a WHITE background and BLACK/gray text, no colored
header band, and no colored sidebar is extremely common and entirely
valid. If the header area is plain white (or matches the page background)
with no visible distinct colored rectangle behind the name/title,
header.bg_color MUST be null — not a guessed or "typical" color. The same
applies to sidebar.bg_color when there is no sidebar panel at all. Only
report a bg_color when you can point to an actual visible rectangle of
that color in the image. When genuinely uncertain whether a decorative
element (colored band, sidebar panel, icon, divider) exists, assume it
does NOT exist rather than inventing one — false positives here directly
produce a generated resume that looks wrong compared to its own template.

IMPORTANT — for any block that has a FILL/BACKGROUND COLOR (a sidebar
panel, a colored header band, etc.), also report whether its height looks
CONTENT-BOUNDED or FULL-COLUMN:
  - "content_bounded": the colored area stops right where its own text
    stops, with plain page background below it (this is extremely common
    in template files because they're filled with short placeholder text
    — the colored box was never meant to end there, it just ran out of
    content to wrap).
  - "full_column": the colored area clearly runs to the bottom edge of the
    page (or would obviously be intended to, e.g. it already reaches
    >90% of page height, or there's a visible border/edge suggesting an
    intentional fixed panel).
When in doubt, prefer "content_bounded" — real resumes are almost always
longer than a template's placeholder text, so a sidebar/header fill should
default to extending with whatever content it actually receives rather
than being capped at the height measured here.

PyMuPDF has extracted these structural hints from the actual PDF bytes.
Treat them as GROUND TRUTH for colors and typography — including when
PyMuPDF reports no color at all (null/none). Only override a null-color
hint if you can clearly see an actual colored rectangle the extraction
missed; never override a null hint with a color merely because it would
look more typical of a resume design.

PYMUPDF HINTS:
{hints_str}

Return ONLY a valid JSON object in this exact shape. No prose, no markdown
fences, and never any template/sample resume content.

{{
  "layout_type": "sidebar-left" | "sidebar-right" | "single-column" | "two-column-balanced",
  "page_margin_pt": {{ "top": 0, "right": 0, "bottom": 0, "left": 0 }},
  "blocks": [
    {{ "name": "header", "x_pct": 0, "y_pct": 0, "width_pct": 100, "height_pct": 16, "height_mode": "content_bounded" | "full_column" | null }},
    {{ "name": "sidebar", "x_pct": 0, "y_pct": 16, "width_pct": 32, "height_pct": 84, "height_mode": "content_bounded" | "full_column" | null }},
    {{ "name": "main", "x_pct": 32, "y_pct": 16, "width_pct": 68, "height_pct": 84, "height_mode": "content_bounded" | "full_column" | null }}
  ],
  "sidebar": {{
    "has_sidebar": true | false,
    "side": "left" | "right" | null,
    "bg_color": "#hex" | null,
    "text_color": "#hex" | null,
    "width_estimate": "30%",
    "padding_pt": {{ "top": 20, "right": 16, "bottom": 20, "left": 16 }},
    "sections": ["Contact", "Skills"]
  }},
  "header": {{
    "bg_color": "#hex" | null,
    "name_color": "#hex",
    "name_size_pt": 24,
    "title_size_pt": 12,
    "padding_pt": {{ "top": 20, "right": 24, "bottom": 20, "left": 24 }},
    "alignment": "left" | "center" | "right",
    "contact_style": "inline-row" | "stacked" | "icon-row",
    "has_photo": true | false,
    "photo_position": "left" | "right" | "center" | null,
    "photo_shape": "circle" | "square" | "rounded" | null,
    "photo_size_pt": 80
  }},
  "section_heading": {{
    "color": "#hex",
    "bg_color": "#hex" | null,
    "uppercase": true | false,
    "bold": true | false,
    "border_bottom": true | false,
    "border_color": "#hex" | null,
    "size_pt": 12,
    "letter_spacing": "normal" | "wide"
  }},
  "body": {{ "color": "#hex", "size_pt": 10 }},
  "accent_color": "#hex",
  "bullet_char": "•" | "▪" | "–" | "→" | "",
  "section_order": ["Summary", "Experience", "Education", "Skills"],
  "spacing": {{
    "section_gap_pt": 14,
    "item_gap_pt": 8,
    "line_height": 1.4,
    "bullet_indent_pt": 14,
    "column_gap_pt": 20
  }},
  "decorative_elements": {{
    "horizontal_dividers": true | false,
    "skill_bars": true | false,
    "icons": true | false,
    "vertical_timeline": true | false
  }}
}}
"""

    content_parts: list = []
    for img_b64 in page_images_b64:
        content_parts.append(
            types.Part(inline_data=types.Blob(mime_type="image/png", data=img_b64))
        )
    if pdf_b64 and not page_images_b64:
        content_parts.append(
            types.Part(inline_data=types.Blob(mime_type="application/pdf", data=pdf_b64))
        )
    content_parts.append(types.Part(text=prompt))

    active_client, active_model = _get_runtime(llm_settings, default_model=GEMINI_MODEL_VISION)
    print(f"[GEMINI] Pass 1: extracting design + geometry tokens from template via vision (model={active_model})...")
    response = active_client.models.generate_content(
        model=active_model,
        contents=content_parts,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=3072,
            response_mime_type="application/json",
        ),
    )

    raw = _collect_text(response)
    print(f"[GEMINI] Pass 1 raw response ({len(raw)} chars): {raw[:300]!r}")

    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 2 — HTML generation (design from TEMPLATE geometry, content from Resume Y)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_html(
    tokens: dict,
    resume_y_text: str,
    pdf_b64: Optional[str],
    page_images_b64: list[str],
    llm_settings: Optional[dict] = None,
    corrections: str = "",
) -> str:
    """
    Generate the final HTML resume. `tokens` (including the geometric
    `blocks`) and the template page image(s) are the VISUAL/DESIGN source.
    `resume_y_text` is the ONLY source of actual resume content.

    `corrections`, when non-empty, is a block of concrete measured
    correction instructions (from template_render_compare.build_corrections_text)
    produced by comparing a PREVIOUS attempt's actual rendered output
    against the template. When present, the model is told to apply ONLY
    those fixes on top of everything else — this is a regeneration pass,
    not a fresh interpretation.
    """

    layout_type = tokens.get("layout_type", "single-column")
    sidebar = tokens.get("sidebar", {}) or {}
    header = tokens.get("header", {}) or {}
    sec_heading = tokens.get("section_heading", {}) or {}
    body_info = tokens.get("body", {}) or {}
    accent = tokens.get("accent_color", "#2c3e50")
    bullet = tokens.get("bullet_char", "•")
    decorative = tokens.get("decorative_elements", {}) or {}
    font_family = tokens.get("font_family") or "Arial, Helvetica, sans-serif"
    spacing = tokens.get("spacing", {}) or {}
    blocks = tokens.get("blocks", []) or []
    margin = tokens.get("page_margin_pt", {}) or {}
    logo = tokens.get("logo", {}) or {}

    section_order = tokens.get("section_order") or tokens.get("sections") or []

    section_gap = spacing.get("section_gap_pt", 14)
    item_gap = spacing.get("item_gap_pt", 8)
    line_height = spacing.get("line_height", 1.4)
    bullet_indent = spacing.get("bullet_indent_pt", 14)
    column_gap = spacing.get("column_gap_pt", 20)

    margin_top = margin.get("top", 0)
    margin_right = margin.get("right", 0)
    margin_bottom = margin.get("bottom", 0)
    margin_left = margin.get("left", 0)

    # ── Layout scaffold from measured geometry — TABLE-based, not Grid ───────
    # CSS Grid/Flexbox support is inconsistent across HTML-to-PDF renderers
    # (many still use older WebKit-based engines with no Grid support at
    # all). A <table> with explicit column widths is the one layout
    # primitive that renders correctly on essentially every engine, so it's
    # what we use for ANY multi-column template — sidebar-left, sidebar-
    # right, 3-column, whatever the uploaded template turns out to be.
    header_block, column_blocks = _detect_columns(blocks)

    has_measured_columns = len(column_blocks) >= 2
    has_sidebar_flag = "sidebar" in layout_type

    if has_measured_columns or has_sidebar_flag:
        # Prefer measured geometry; fall back to the qualitative sidebar
        # tokens only if geometry extraction didn't find a clean column set
        if has_measured_columns:
            widths = _table_widths_pct(column_blocks)
        else:
            sidebar_width_num = 32.0
            try:
                sidebar_width_num = float(str(sidebar.get("width_estimate", "32%")).rstrip("%"))
            except ValueError:
                pass
            sidebar_side_fallback = sidebar.get("side", "left")
            widths = (
                [sidebar_width_num, 100 - sidebar_width_num]
                if sidebar_side_fallback == "left"
                else [100 - sidebar_width_num, sidebar_width_num]
            )

        sidebar_bg = sidebar.get("bg_color") or accent
        sidebar_text = sidebar.get("text_color") or "#ffffff"
        sidebar_side = sidebar.get("side", "left")
        sidebar_pad = _clamp_padding(sidebar.get("padding_pt", {}) or {}, label="sidebar")
        sidebar_sections = sidebar.get("sections", []) or []
        main_sections = [s for s in section_order if s not in sidebar_sections]

        # Which measured column index is the sidebar? Leftmost column if
        # side==left, rightmost if side==right (matches how blocks were sorted).
        n_cols = len(widths)
        sidebar_index = 0 if sidebar_side == "left" else n_cols - 1

        td_specs = []
        for i, w in enumerate(widths):
            if i == sidebar_index:
                pad = (
                    f'{sidebar_pad.get("top",24)}pt {sidebar_pad.get("right",18)}pt '
                    f'{sidebar_pad.get("bottom",24)}pt {sidebar_pad.get("left",18)}pt'
                )
                td_specs.append({
                     "role": "sidebar",
                   "width": w,
                   "style": (
                             f'background:{sidebar_bg}; color:{sidebar_text}; padding:{pad}; '
                             f'box-decoration-break: clone; -webkit-box-decoration-break: clone; '
                             f'min-height: 297mm;'
                             ),
                   "sections": sidebar_sections or ["Contact", "Skills", "Languages"],
                })
            else:
                td_specs.append({
                    "role": "main",
                    "width": w,
                    "style": f'background:#ffffff; color:{body_info.get("color","#333333")}; padding:24pt 22pt;',
                    "sections": main_sections or [s for s in section_order if s],
                })

        col_cells = "\n".join(
            f'    <td width="{spec["width"]}%" class="{spec["role"]}-col" style="{spec["style"]}">\n'
            f'      <!-- {spec["role"].upper()} column sections, in this exact order: '
            f'{", ".join(spec["sections"]) or "infer from Resume Y"} -->\n'
            f'    </td>'
            for spec in td_specs
        )

        header_row_markup = ""
        if header_block is not None:
            header_row_markup = f"""
Before the table, render a FULL-WIDTH header row (a <div>, not part of the
table), matching the template's measured header block:
  <div class="header-block">
    <!-- name, title, and any header-level content here -->
  </div>
This header sits ABOVE the columns and spans the full page width — it is
not inside either column.
"""
        else:
            header_row_markup = """
The template has NO separate full-width header band — the name/title sit
as the first item inside whichever column contained them in the template
image (commonly the sidebar for left-sidebar templates, or the main
column for right-sidebar templates). Do not invent a full-width header
band if the template doesn't have one.
"""

        layout_instruction = f"""
LAYOUT: {n_cols}-column table layout, widths measured directly from the
template image (do not deviate from these numbers, do not let a column
resize based on content length):

  Column widths (left to right): {', '.join(f'{w}%' for w in widths)}

Build the columns using EXACTLY this structure — a single <table>, one
<tr>, one <td> per column, each <td> carrying its own background/padding
inline as shown. Do NOT use display:grid or display:flex for this
structural layout — use only the table below (grid/flex support varies
across the rendering engines this HTML may be processed by, tables do
not):

<table class="resume-layout" role="presentation" cellpadding="0" cellspacing="0" style="width:100%; border-collapse:collapse; table-layout:fixed;">
  <tr>
{col_cells}
  </tr>
</table>
{header_row_markup}
CRITICAL — column fill color height: IGNORE any measured height for a
column's background color. The template's own placeholder text is almost
certainly a different length than Resume Y's real content, so a height
measured off the template image would be wrong here — using it is exactly
what causes a sidebar's color to stop partway down the page while its
text keeps going below it (a real, verified failure mode, not a
theoretical one). Instead: every column that has a background color in
the template MUST have that color fill its own actual rendered height in
THIS output — full page height at minimum (min-height:297mm inline on
that <td>), and taller still if that column's content pushes onto a
second page. Column WIDTH, x-position, padding, and color are still taken
exactly from the measurements above — only fill height is content-driven,
never template-measured.
"""
    else:
        layout_instruction = f"""
LAYOUT: Single column, measured page margins top:{margin_top}pt right:{margin_right}pt
bottom:{margin_bottom}pt left:{margin_left}pt. Use a centered .page-wrapper div,
max-width 800px. All sections flow vertically in order: {', '.join(section_order) or 'as inferred from the template'}.
"""

    # ── Heading CSS ───────────────────────────────────────────────────────────
    heading_color = sec_heading.get("color") or accent
    heading_bg = sec_heading.get("bg_color")
    heading_uppercase = sec_heading.get("uppercase", False)
    heading_bold = sec_heading.get("bold", True)
    heading_border = sec_heading.get("border_bottom", True)
    heading_border_c = sec_heading.get("border_color") or accent

    heading_css_rules = f"""
    color: {heading_color};
    {'background: ' + heading_bg + '; padding: 4px 8px;' if heading_bg else ''}
    {'text-transform: uppercase; letter-spacing: 1px;' if heading_uppercase else ''}
    font-weight: {'bold' if heading_bold else 'normal'};
    font-size: {sec_heading.get('size_pt', 12)}pt;
    {'border-bottom: 2px solid ' + heading_border_c + '; padding-bottom: 4px;' if heading_border else ''}
    margin-bottom: {item_gap}pt;
    margin-top: {section_gap}pt;
    display: block;
    page-break-after: avoid;
"""

    # ── Header CSS ─────────────────────────────────────────────────────────────
    header_bg = header.get("bg_color")
    name_color = header.get("name_color") or ("#ffffff" if header_bg else "#000000")
    name_size = header.get("name_size_pt", 24)
    header_pad = _clamp_padding(header.get("padding_pt", {}) or {}, label="header")
    header_align = header.get("alignment", "left")
    contact_style = header.get("contact_style", "inline-row")

    if contact_style == "stacked":
        contact_instruction = "Stack each contact item on its own line."
    elif contact_style == "icon-row":
        contact_instruction = "Place contact items in one row, each prefixed by a small unicode icon."
    else:
        contact_instruction = "Place contact items in a single horizontal row separated by  |  or a bullet."

    header_padding_css = (
        f"{header_pad.get('top',20)}pt {header_pad.get('right',24)}pt "
        f"{header_pad.get('bottom',20)}pt {header_pad.get('left',24)}pt"
    )

    # ── Skill-bar instruction ─────────────────────────────────────────────────
    skill_bar_instruction = ""
    if decorative.get("skill_bars"):
        skill_bar_instruction = f"""
For skill levels, render a table-based progress bar:
  <table width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse; margin:3px 0;">
    <tr>
      <td style="width:80%; height:6px; background:{accent};"></td>
      <td style="background:#cccccc; height:6px;"></td>
    </tr>
  </table>
Set the first td's width inline to reflect the proficiency level.
"""

    # ── Logo instruction (placement-aware — see _build_logo_instruction) ────
    logo_instruction = _build_logo_instruction(logo)

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt_lines = [
        "You are an expert resume formatter and HTML/CSS engineer.",
        "Your output will be rendered to PDF by a real Chromium engine.",
        "",
        "═" * 60,
        "WHAT THE REFERENCE FILE IS",
        "═" * 60,
        "",
        "The attached page image(s) are a RESUME TEMPLATE, used ONLY as a",
        "visual/design reference. Any text visible in it is sample/placeholder",
        "content and MUST NOT be copied into the output under any circumstance.",
        "",
        "The measurements in STEP 1 below were taken directly off this",
        "template image (bounding boxes, padding, spacing). Treat them as",
        "hard constraints, not suggestions — do not round them to something",
        "that 'looks about right'; use the exact numbers given.",
        "",
        "The actual resume content comes exclusively from Resume Y in STEP 4.",
        "",
        "═" * 60,
        "OBJECTIVE",
        "═" * 60,
        "",
        "Reproduce the TEMPLATE's measured visual design exactly, filled with",
        "the complete content of Resume Y.",
        "",
        "COPY FROM THE TEMPLATE (visual/structural only, using the exact",
        "measurements given — do not approximate):",
        "  • column widths and positions (see grid spec below)",
        "  • page margins, header padding, sidebar padding",
        "  • section gap / item gap / line-height / bullet indent (see spacing)",
        "  • colors, header design, typography, heading style",
        "  • bullet style, borders, dividers, decorative elements",
        "  • photo placement and shape",
        "  • the logo mark, via placeholder only, AT ITS MEASURED POSITION — see LOGO instruction below",
        "",
        "DO NOT COPY FROM THE TEMPLATE (any factual/sample content):",
        "  • names, titles, emails, phones, addresses, companies, dates,",
        "    descriptions, projects, skills, education, or any other sample text",
        "",
        "═" * 60,
        "STEP 1 — TARGET LAYOUT (measured, not estimated)",
        "═" * 60,
        layout_instruction,
        f"Spacing constants to use EXACTLY: section-gap={section_gap}pt, "
        f"item-gap={item_gap}pt, line-height={line_height}, "
        f"bullet-indent={bullet_indent}pt, column-gap={column_gap}pt.",
        "",
        "═" * 60,
        "STEP 1B — LOGO",
        "═" * 60,
        logo_instruction,
        "",
        "═" * 60,
        "STEP 2 — MANDATORY CSS (place verbatim FIRST inside <style>)",
        "═" * 60,
        "",
        "<style>",
        REQUIRED_CSS,
        "",
        "/* === CUSTOM STYLES BELOW — use the exact measured values === */",
        "",
        f"body {{ font-family: {font_family}; line-height: {line_height}; }}",
        "",
        "h2, .section-title {",
        heading_css_rules,
        "}",
        "",
        ".sidebar-col h2, .sidebar-col .section-title {",
        f"    color: {sidebar.get('text_color', '#ffffff') if sidebar.get('has_sidebar') else heading_color};",
        f"    border-bottom-color: {sidebar.get('text_color', '#ffffff') if sidebar.get('has_sidebar') else heading_border_c};",
        "}",
        "",
        ".header-block {",
        f"    padding: {header_padding_css};",
        f"    text-align: {header_align};",
        f"    {'background: ' + header_bg + ';' if header_bg else ''}",
        "}",
        "",
        ".resume-name {",
        f"    font-size: {name_size}pt;",
        "    font-weight: bold;",
        f"    color: {name_color};",
        "    line-height: 1.15;",
        f"    margin-bottom: {max(item_gap - 2, 2)}pt;",
        "    overflow-wrap: break-word;",
        "    word-break: break-word;",
        "    hyphens: auto;",
        "    max-width: 100%;",
        "}",
        "",
        ".resume-title {",
        f"    font-size: {header.get('title_size_pt', 11)}pt;",
        f"    color: {name_color};",
        "    opacity: 0.9;",
        f"    margin-bottom: {item_gap}pt;",
        "    overflow-wrap: break-word;",
        "    word-break: break-word;",
        "}",
        "",
        ".sidebar-col, .sidebar-col * {",
        "    overflow-wrap: break-word;",
        "    word-break: break-word;",
        "}",
        "",
        ".bullet {",
        f"    padding-left: {bullet_indent}pt;",
        f"    text-indent: -{max(bullet_indent - 4, 4)}pt;",
        "    margin: 3px 0;",
        "    page-break-inside: avoid;",
        "}",
        "",
        ".job-title {",
        "    font-weight: bold;",
        f"    color: {heading_color};",
        "    font-size: 10.5pt;",
        "}",
        ".job-meta {",
        "    font-size: 9pt;",
        "    color: #666666;",
        f"    margin-bottom: {max(item_gap - 3, 2)}pt;",
        "    font-style: italic;",
        "}",
        "",
        ".tpl-logo {",
        "    margin-bottom: 6pt;",
        "}",
        "</style>",
        "",
        "═" * 60,
        "STEP 3 — RENDERER CAPABILITIES",
        "═" * 60,
        "",
        "This HTML may be rendered by different engines depending on where",
        "it's used, including older ones with limited modern-CSS support. You MAY use:",
        "  ✓ the <table class='resume-layout'> scaffold from STEP 1 — the ONLY",
        "    mechanism for side-by-side columns, full stop",
        "  ✓ border-radius, box-shadow, basic color/typography CSS on individual elements",
        "  ✓ SVG icons inline",
        "  ✓ prefer <div class='bullet'> over <ul>/<li> for tighter spacing control",
        "  ✓ pt, px, %, em, rem units",
        "  ✓ inline styles where needed for precise placement",
        f"  ✓ the single exact <img src=\"{LOGO_PLACEHOLDER_TOKEN}\"> tag described in STEP 1B, if and only if a logo was detected",
        "",
        "Do NOT use, anywhere in the document:",
        "  ✗ display:grid or display:flex for structural/column layout — use the",
        "    table scaffold from STEP 1 instead, even if grid/flex feels simpler",
        "  ✗ CSS transforms, position:absolute/fixed for layout purposes",
        "  ✗ External fonts/Google Fonts (no @import — keep offline)",
        "  ✗ JavaScript (none will execute)",
        "  ✗ Images referenced by external URL (no network at render time)",
        f"  ✗ Any <img> tag other than the exact placeholder from STEP 1B — never",
        "    invent your own logo/icon image, and never modify the placeholder's src value",
        "",
        "═" * 60,
        "STEP 4 — CONTENT (Resume Y is the ONLY source of resume content)",
        "═" * 60,
        "",
        "Resume Y — CONTENT SOURCE (use ALL of this, do not truncate):",
        "─" * 40,
        resume_y_text,
        "",
        "═" * 60,
        "STEP 5 — GENERATION RULES",
        "═" * 60,
        "",
        "Header block:",
        "  • Wrap the header in <div class='header-block'>...</div>",
        f"  • Render full name as <div class='resume-name'>NAME</div>",
        "  • CRITICAL — the sidebar/column WIDTH is fixed and measured; it must",
        "    never be widened or shrunk to accommodate a long name. Instead the",
        "    NAME TEXT must be sized to fit the fixed column. Do not insert a",
        "    manual <br> to force a line split between name parts (e.g. between",
        "    first and last name) — let the browser wrap naturally based on",
        "    spaces, unless the template image itself clearly shows a fixed,",
        "    deliberate line break at that exact point.",
        f"  • Font-size scaling for the name, applied deterministically based on",
        f"    the full name's character count (including spaces), starting from",
        f"    the measured size of {name_size}pt:",
        f"      - under 14 characters: use {name_size}pt as measured",
        f"      - 14-20 characters: use {round(name_size * 0.8, 1)}pt",
        f"      - 21-28 characters: use {round(name_size * 0.65, 1)}pt",
        f"      - over 28 characters: use {round(name_size * 0.55, 1)}pt",
        "    Apply this as an inline font-size override on that specific",
        "    .resume-name element if it differs from the CSS default above —",
        "    this is the ONE property allowed to override the class default.",
        f"  • {contact_instruction}",
        f"  • If a logo placeholder is used (per STEP 1B), place it EXACTLY at",
        "    the position described in STEP 1B — do not default to a generic",
        "    location if STEP 1B gave you a specific zone/position.",
        "",
        "Content rules:",
        "  • Use Resume Y content ONLY. Never copy any text from the template.",
        "  • Include EVERY section, EVERY bullet point from Resume Y — do not truncate.",
        f"  • Bullet character: '{bullet}' (not - or *).",
        f"  • Section order: {section_order or 'infer a sensible order from Resume Y'}",
        "  • If Resume Y has a section not in the template, add it using the",
        "    template's existing visual language (same heading/spacing rules).",
        "  • If the template has a section Resume Y lacks, omit it — never invent content.",
        "  • If Resume Y is longer than one page, add additional pages that repeat",
        "    the same measured layout (same margins/columns/spacing) — never",
        "    truncate content to force a single page.",
        "",
        "Section wrapper:",
        "  <div class='section'>",
        "    <h2 class='section-title'>SECTION NAME</h2>",
        "    ...content...",
        "  </div>",
        "",
        "Job entry:",
        "  <div class='job-entry'>",
        "    <div class='job-title'>Job Title</div>",
        "    <div class='job-meta'>Company &nbsp;|&nbsp; City &nbsp;|&nbsp; Jan 2020 – Dec 2022</div>",
        f"    <div class='bullet'>{bullet}&nbsp; First achievement.</div>",
        f"    <div class='bullet'>{bullet}&nbsp; Second achievement.</div>",
        "  </div>",
        "",
        "Education entry:",
        "  <div class='edu-item'>",
        "    <div class='job-title'>Degree, Field</div>",
        "    <div class='job-meta'>Institution &nbsp;|&nbsp; Year</div>",
        "  </div>",
        "",
        skill_bar_instruction,
        "",
    ]

    if corrections:
        prompt_lines += [
            "═" * 60,
            "STEP 6 — CORRECTIONS FROM PREVIOUS ATTEMPT (measured, not guessed)",
            "═" * 60,
            "",
            "A previous attempt at this exact same template + Resume Y was actually",
            "rendered to an image and measured pixel-by-pixel against the template.",
            "The following differences were detected programmatically (not by",
            "eyeballing) and must be corrected in this new attempt:",
            "",
            corrections,
            "",
            "Everything else in this prompt (layout, colors, content, spacing not",
            "mentioned above) is already correct or unrelated — do not change it",
            "just because you're regenerating. This is a targeted fix, not a redesign.",
            "",
        ]

    prompt_lines += [
        "═" * 60,
        "OUTPUT",
        "═" * 60,
        "",
        "Return ONLY the complete HTML document:",
        "  • Start with <!DOCTYPE html>",
        "  • End with </html>",
        "  • No markdown fences, no explanation, no preamble",
        "  • <style> must include REQUIRED_CSS FIRST, then custom styles",
    ]

    prompt = "\n".join(prompt_lines)

    text_provider = _selected_text_provider(llm_settings)
    if text_provider != "gemini":
        client, provider, active_model, api_key = get_text_llm_runtime(llm_settings)
        print(f"[GEMINI] Pass 2: generating HTML via {provider} model={active_model} (text-only)")
        html_content = generate_llm_text(
            provider=provider,
            client=client,
            model=active_model,
            api_key=api_key,
            prompt=prompt,
            system_instruction=(
                "You are an expert resume formatter and HTML/CSS engineer. "
                "Return only the final HTML document."
            ),
            temperature=0.1,
            max_output_tokens=32768,
        )
        html_content = _clean_html(html_content)
        html_content = _ensure_doctype(html_content)
        _validate_html(html_content, "TEXT_PROVIDER")
        return html_content

    content_parts: list = []
    for img_b64 in page_images_b64:
        content_parts.append(
            types.Part(inline_data=types.Blob(mime_type="image/png", data=img_b64))
        )
    if pdf_b64 and not page_images_b64:
        content_parts.append(
            types.Part(inline_data=types.Blob(mime_type="application/pdf", data=pdf_b64))
        )
    content_parts.append(types.Part(text=prompt))

    active_client, active_model = _get_runtime(llm_settings, default_model=GEMINI_MODEL_HTML)
    print(f"[GEMINI] Pass 2: generating HTML (model={active_model}, template images={len(page_images_b64)})")

    response = active_client.models.generate_content(
        model=active_model,
        contents=content_parts,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=32768,
        ),
    )

    finish_reason = "UNKNOWN"
    if response.candidates:
        finish_reason = str(response.candidates[0].finish_reason)
        if "MAX_TOKENS" in finish_reason:
            print(f"[GEMINI] WARNING: output truncated at token limit")

    html_content = _collect_text(response)
    html_content = _clean_html(html_content)
    html_content = _ensure_doctype(html_content)
    _validate_html(html_content, finish_reason)

    print(f"[GEMINI] Returning validated HTML ({len(html_content)} chars)")
    return html_content


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 3 (OPTIONAL) — Render-and-compare verification loop
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the biggest remaining lever for true pixel fidelity, and it's
# intentionally left as a stub because it needs to call into whatever
# HTML→image/PDF renderer your app already uses (you clearly have one,
# since the final output gets rendered to PDF via Chromium/Playwright).
#
# The pattern:
#   1. render_html_to_png(html_content) -> generated_png_bytes
#   2. send [template_page_image, generated_png] to Gemini together
#   3. ask for a JSON list of concrete deltas: block name, expected vs
#      actual position/size/color, in the same units as the token schema
#   4. if deltas are non-trivial, re-run _generate_html() with an extra
#      "CORRECTIONS" section appended to the prompt listing those deltas
#   5. cap at 1-2 repair iterations to bound cost/latency
#
# def _verify_and_repair(html_content, template_page_images_b64, tokens,
#                         resume_y_text, llm_settings=None, max_iterations=1):
#     for _ in range(max_iterations):
#         generated_png_b64 = your_render_to_png_function(html_content)  # TODO: wire this up
#         diff = _diff_against_template(template_page_images_b64[0], generated_png_b64, llm_settings)
#         if not diff.get("significant_issues"):
#             break
#         html_content = _generate_html(
#             tokens=tokens, resume_y_text=resume_y_text,
#             pdf_b64=None, page_images_b64=template_page_images_b64,
#             llm_settings=llm_settings,
#             # extra_corrections=diff["issues"]  # thread this into the prompt
#         )
#     return html_content


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_text(response) -> str:
    parts = []
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def _clean_html(html: str) -> str:
    html = html.replace("```html", "").replace("```", "").strip()
    return html


def _ensure_doctype(html: str) -> str:
    """If Gemini returned only a fragment, wrap it into a full HTML doc."""
    stripped = html.lstrip().lower()
    if stripped.startswith(("<!doctype", "<html")):
        return html
    wrapped = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "<meta charset='UTF-8'>\n"
        "<title>Resume</title>\n"
        f"<style>{REQUIRED_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{html}\n"
        "</body>\n"
        "</html>"
    )
    print("[GEMINI] _ensure_doctype: wrapped fragment into full HTML document")
    return wrapped


def _validate_html(html: str, finish_reason: str) -> None:
    if len(html) < 500:
        raise ValueError(
            f"Gemini returned too little content ({len(html)} chars). "
            f"Finish reason: {finish_reason}."
        )
    if not html.lstrip().lower().startswith(("<!doctype", "<html")):
        raise ValueError(f"Gemini did not return HTML. Finish reason: {finish_reason}.")
    if "@page" not in html:
        print("[GEMINI] WARNING: @page rule missing")
    if "page-break-inside" not in html:
        print("[GEMINI] WARNING: no page-break rules found")