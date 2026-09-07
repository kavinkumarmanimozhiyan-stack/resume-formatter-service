
"""
pdf_generator.py
─────────────────────────────────────────────────────────────────────────────
Converts an HTML string to a PDF.

Major change: Playwright/Chromium is now the PRIMARY renderer.
wkhtmltopdf is kept as a fallback only for environments where Playwright
is not installed.

Why Playwright:
  • Real Chromium → supports flexbox, grid, modern CSS, web fonts, SVG
  • Honors @page rules correctly on every page
  • No silent CSS failures (the #1 problem with wkhtmltopdf for resumes)

Install once:
    pip install playwright
    playwright install chromium
"""

import os
import re
import uuid
import shutil
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Safety-net CSS (used by both renderers)
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_CSS = """
/* ══ SAFETY NET — injected by pdf_generator ══ */
@page {
    margin: 0;
    size: A4 portrait;
}
* { box-sizing: border-box; }

.section, .resume-section, .section-block, .job, .job-entry,
.experience-item, .edu-item, .education-item, .cert-item,
.project-item, .skill-group {
    page-break-inside: avoid;
}
tr { page-break-after: auto; }
table.resume-layout > tbody > tr { page-break-inside: auto; }
h1, h2, h3, h4 { page-break-after: avoid; }
p, .bullet, .bullet-item {
    page-break-inside: avoid;
    orphans: 2; widows: 2;
}
table.resume-layout td {
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}
td.sidebar-col { page-break-inside: avoid; }
/* ══ END SAFETY NET ══ */
"""

def _inject_safety_css(html: str) -> str:
    injection = f"\n/* injected safety CSS */\n{SAFETY_CSS}\n"
    last_style_end = html.rfind("</style>")
    if last_style_end != -1:
        return html[:last_style_end] + injection + html[last_style_end:]
    head_end = html.lower().find("</head>")
    if head_end != -1:
        style_tag = f"\n<style>\n{SAFETY_CSS}\n</style>\n"
        return html[:head_end] + style_tag + html[head_end:]
    return f"<style>\n{SAFETY_CSS}\n</style>\n" + html


# ─────────────────────────────────────────────────────────────────────────────
#  Renderer: Playwright (PRIMARY)
# ─────────────────────────────────────────────────────────────────────────────

def _render_with_playwright(html: str, output_path: str) -> bool:
    """Render HTML to PDF using Playwright/Chromium. Returns True on success."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[PDF_GENERATOR] Playwright not installed — falling back")
        return False

    print("[PDF_GENERATOR] Rendering with Playwright/Chromium...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context()
            page = context.new_page()
            page.set_content(html, wait_until="networkidle")
            page.emulate_media(media="print")
            page.pdf(
              path=output_path,
              format="A4",
               print_background=True,
    margin={
        "top":    "0mm",
        "right":  "0mm",
        "bottom": "0mm",
        "left":   "0mm",
    },
    prefer_css_page_size=True,
)
            browser.close()
        print(f"[PDF_GENERATOR] Playwright render OK")
        return True
    except Exception as e:
        print(f"[PDF_GENERATOR] Playwright render failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Renderer: wkhtmltopdf (FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────

def _get_wkhtmltopdf_config():
    import pdfkit
    env_path = os.environ.get("WKHTMLTOPDF_PATH")
    if env_path and os.path.isfile(env_path):
        return pdfkit.configuration(wkhtmltopdf=env_path)
    if shutil.which("wkhtmltopdf"):
        return None
    win_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    if os.path.isfile(win_path):
        return pdfkit.configuration(wkhtmltopdf=win_path)
    return None


def _render_with_wkhtmltopdf(html: str, output_path: str) -> bool:
    try:
        import pdfkit
    except ImportError:
        return False

    print("[PDF_GENERATOR] Rendering with wkhtmltopdf (fallback)...")
    try:
        options = {
            "page-size": "A4",
            "orientation": "Portrait",
            "margin-top":    "15mm",
            "margin-right":  "10mm",
            "margin-bottom": "15mm",
            "margin-left":   "10mm",
            "encoding": "UTF-8",
            "print-media-type": "",
            "disable-smart-shrinking": "",
            "zoom": "1",
            "dpi":  "96",
            "enable-local-file-access": "",
            "no-outline": "",
            "quiet": "",
        }
        config = _get_wkhtmltopdf_config()
        pdfkit.from_string(html, output_path, options=options, configuration=config)
        return True
    except Exception as e:
        print(f"[PDF_GENERATOR] wkhtmltopdf render failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_from_html(html_content: str, output_dir: str) -> str:
    """
    Convert HTML to PDF.

    Renderer priority:
      1. Playwright/Chromium (primary — full modern CSS support)
      2. wkhtmltopdf         (fallback — limited but usable)
    """
    print(f"[PDF_GENERATOR] generate_pdf_from_html called")
    print(f"[PDF_GENERATOR] HTML input: {len(html_content)} chars")
    print(f"[PDF_GENERATOR] Output dir: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    html_content = _inject_safety_css(html_content)
    print(f"[PDF_GENERATOR] After CSS injection: {len(html_content)} chars")

    filename    = f"resume_{uuid.uuid4().hex}.pdf"
    output_path = os.path.abspath(os.path.join(output_dir, filename))
    print(f"[PDF_GENERATOR] Output path: {output_path}")

    # Try Playwright first
    ok = _render_with_playwright(html_content, output_path)
    if not ok:
        ok = _render_with_wkhtmltopdf(html_content, output_path)

    if not ok or not os.path.exists(output_path):
        raise RuntimeError(
            "PDF generation failed. Install Playwright "
            "(`pip install playwright && playwright install chromium`) "
            "or wkhtmltopdf (`apt install wkhtmltopdf`)."
        )

    file_size = os.path.getsize(output_path)
    if file_size < 1000:
        raise RuntimeError(
            f"PDF at {output_path} is suspiciously small ({file_size} bytes)."
        )

    print(f"[PDF_GENERATOR] PDF generated: {output_path} ({file_size:,} bytes)")
    return output_path
