"""
resume_controller.py
─────────────────────────────────────────────────────────────────────────────
FastAPI controller for the resume formatting endpoints.

STORAGE POLICY:
  Resume uploads (format resume, content resume), DOCX→PDF conversions, and
  generated output PDFs are NEVER written to a project-local folder. Each
  request gets its own directory under the OS temp location
  (tempfile.mkdtemp()), and that ENTIRE directory is removed once the
  request is done — for JSON responses, in a `finally` block; for file
  (PDF) responses, via a BackgroundTask that runs after the file has been
  streamed to the client.

  Templates are the only thing intentionally persisted, and only in
  Cloudinary (see template_store.py) — never on local disk.

WORD OUTPUT POLICY (added):
  The "Word download" path (output_format in {"html", "docx"}) now produces
  a REAL binary .docx via LibreOffice headless, converted from the exact
  same HTML that the PDF path renders — not a raw-HTML-wrapped-as-.doc
  trick. This mirrors the existing _convert_docx_to_pdf pattern used
  elsewhere in this file (same LibreOffice-discovery logic, same
  subprocess/timeout handling), just in the opposite conversion direction.
  If LibreOffice is unavailable or the conversion fails for any reason,
  the pipeline falls back to returning the raw HTML (as before) so the
  endpoint never hard-fails just because DOCX conversion isn't available
  in a given environment.
"""

import asyncio
import base64
import os
import shutil
import sys
import tempfile
import uuid
import subprocess
from datetime import datetime, timezone
from typing import Optional

from fastapi import UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from app.services.document_parser import extract_text_from_document
from app.services.gemini_service import generate_formatted_resume
from app.services.pdf_generator import generate_pdf_from_html
from app.config import settings


# ─────────────────────────────────────────────────────────────────────────────
#  File-type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_file_extension(file: UploadFile, content: bytes) -> str:
    if content[:4] == b"%PDF":
        return ".pdf"
    if content[:2] == b"PK":
        return ".docx"
    if content[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return ".doc"

    if file.filename:
        name = file.filename.lower()
        for ext in (".pdf", ".docx", ".doc"):
            if name.endswith(ext):
                return ext

    ct = (file.content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "wordprocessingml" in ct or "msword" in ct:
        return ".docx"

    return ".pdf"


def _is_valid_resume_file(file: UploadFile, content: bytes) -> bool:
    return _get_file_extension(file, content) in (".pdf", ".docx", ".doc")


# ─────────────────────────────────────────────────────────────────────────────
#  LibreOffice discovery (shared by DOCX→PDF and HTML→DOCX conversion)
# ─────────────────────────────────────────────────────────────────────────────

def _find_libreoffice() -> Optional[str]:
    for env_name in ("LIBREOFFICE_PATH", "SOFFICE_PATH"):
        env_path = os.environ.get(env_name)
        if env_path and os.path.isfile(env_path):
            return env_path
    for command in ("libreoffice", "soffice"):
        exe = shutil.which(command)
        if exe:
            return exe
    for candidate in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        r"C:\Program Files\LibreOffice\program\libreoffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\libreoffice.exe",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  DOCX → PDF conversion (for Gemini Vision)
# ─────────────────────────────────────────────────────────────────────────────

def _convert_docx_to_pdf(docx_path: str) -> Optional[str]:
    """Convert a .docx/.doc file to PDF, writing alongside it in the same temp dir."""
    out_dir = os.path.abspath(os.path.dirname(docx_path))
    docx_path = os.path.abspath(docx_path)
    pdf_path = os.path.splitext(docx_path)[0] + ".pdf"

    def _pdf_is_ready(path: str) -> bool:
        return os.path.exists(path) and os.path.getsize(path) > 1000

    office_exe = _find_libreoffice()
    if office_exe:
        try:
            print(f"[CONTROLLER] Converting DOCX to PDF with LibreOffice: {office_exe}")
            result = subprocess.run(
                [office_exe, "--headless", "--convert-to", "pdf", "--outdir", out_dir, docx_path],
                capture_output=True, text=True, timeout=30,
            )
            if _pdf_is_ready(pdf_path):
                print(f"[CONTROLLER] DOCX to PDF via LibreOffice: {pdf_path}")
                return pdf_path
            print(f"[CONTROLLER] LibreOffice conversion failed or produced no PDF; stderr: {result.stderr}")
        except FileNotFoundError:
            print("[CONTROLLER] LibreOffice not available at configured path")
        except subprocess.TimeoutExpired:
            print("[CONTROLLER] LibreOffice conversion timed out (30s)")
        except Exception as e:
            print(f"[CONTROLLER] LibreOffice conversion failed: {e}")

    try:
        import importlib.util
        if importlib.util.find_spec("win32com"):
            print("[CONTROLLER] Converting DOCX to PDF with direct Microsoft Word COM...")
            result = subprocess.run(
                [
                    sys.executable, "-c",
                    (
                        "import os, sys, pythoncom, win32com.client; "
                        "docx, pdf = sys.argv[1], sys.argv[2]; "
                        "pythoncom.CoInitialize(); "
                        "word = win32com.client.DispatchEx('Word.Application'); "
                        "word.Visible = False; "
                        "doc = word.Documents.Open(docx, ReadOnly=True, AddToRecentFiles=False); "
                        "doc.ExportAsFixedFormat(pdf, 17); "
                        "doc.Close(False); "
                        "\ntry:\n    word.Quit()\nexcept Exception:\n    pass\n"
                        "pythoncom.CoUninitialize(); "
                        "sys.exit(0 if os.path.exists(pdf) and os.path.getsize(pdf) > 1000 else 1)"
                    ),
                    docx_path, pdf_path,
                ],
                capture_output=True, text=True, timeout=30,
            )
            if _pdf_is_ready(pdf_path):
                print(f"[CONTROLLER] DOCX to PDF via direct Word COM: {pdf_path}")
                return pdf_path
            print(f"[CONTROLLER] direct Word COM conversion failed; stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("[CONTROLLER] direct Word COM conversion timed out (30s)")
    except Exception as e:
        print(f"[CONTROLLER] direct Word COM unavailable/failed: {e}")

    try:
        import importlib.util
        if os.environ.get("ENABLE_DOCX2PDF", "").strip().lower() in ("1", "true", "yes") and importlib.util.find_spec("docx2pdf"):
            print("[CONTROLLER] Converting DOCX to PDF with docx2pdf / Microsoft Word...")
            result = subprocess.run(
                [sys.executable, "-c", "import sys; from docx2pdf import convert; convert(sys.argv[1], sys.argv[2])", docx_path, pdf_path],
                capture_output=True, text=True, timeout=20,
            )
            if _pdf_is_ready(pdf_path):
                print(f"[CONTROLLER] DOCX to PDF via docx2pdf: {pdf_path}")
                return pdf_path
            print(f"[CONTROLLER] docx2pdf conversion failed. stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("[CONTROLLER] docx2pdf conversion timed out (20s)")
    except Exception as e:
        print(f"[CONTROLLER] docx2pdf unavailable/failed: {e}")

    if not office_exe:
        print("[CONTROLLER] No DOCX to PDF converter found. Install LibreOffice / set LIBREOFFICE_PATH.")
        return None

    try:
        print(f"[CONTROLLER] Converting DOCX to PDF with: {office_exe}")
        result = subprocess.run(
            [office_exe, "--headless", "--convert-to", "pdf", "--outdir", out_dir, docx_path],
            capture_output=True, text=True, timeout=45,
        )
        if result.returncode != 0:
            print(f"[CONTROLLER] LibreOffice stderr: {result.stderr}")
            return None
        if os.path.exists(pdf_path):
            print(f"[CONTROLLER] DOCX→PDF: {pdf_path}")
            return pdf_path
        print(f"[CONTROLLER] LibreOffice ran but PDF not found at: {pdf_path}")
        return None
    except FileNotFoundError:
        print("[CONTROLLER] LibreOffice not installed — skipping DOCX→PDF conversion")
        return None
    except subprocess.TimeoutExpired:
        print("[CONTROLLER] LibreOffice conversion timed out (45s)")
        return None
    except Exception as e:
        print(f"[CONTROLLER] DOCX→PDF unexpected error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  HTML → DOCX conversion (Word download path — produces a REAL .docx)
# ─────────────────────────────────────────────────────────────────────────────

def _convert_html_to_docx_libreoffice(html_path: str, work_dir: str, docx_path: str) -> Optional[str]:
    """LibreOffice headless HTML→DOCX. Preferred when available (works on
    Windows, macOS, and Linux, so it's the right choice for production)."""
    office_exe = _find_libreoffice()
    if not office_exe:
        print("[CONTROLLER] No LibreOffice found for HTML→DOCX conversion.")
        return None

    try:
        print(f"[CONTROLLER] Converting HTML to DOCX with LibreOffice: {office_exe}")
        result = subprocess.run(
            [office_exe, "--headless", "--convert-to", "docx", "--outdir", work_dir, html_path],
            capture_output=True, text=True, timeout=45,
        )
        if os.path.exists(docx_path) and os.path.getsize(docx_path) > 1000:
            print(f"[CONTROLLER] HTML→DOCX via LibreOffice: {docx_path}")
            return docx_path
        print(f"[CONTROLLER] LibreOffice HTML→DOCX conversion failed or produced no DOCX; stderr: {result.stderr}")
        return None
    except FileNotFoundError:
        print("[CONTROLLER] LibreOffice not available at configured path for HTML→DOCX")
        return None
    except subprocess.TimeoutExpired:
        print("[CONTROLLER] LibreOffice HTML→DOCX conversion timed out (45s)")
        return None
    except Exception as e:
        print(f"[CONTROLLER] LibreOffice HTML→DOCX unexpected error: {e}")
        return None


def _convert_html_to_docx_word_com(html_path: str, docx_path: str) -> Optional[str]:
    """
    Microsoft Word COM automation fallback for HTML→DOCX, for Windows dev/
    staging machines that have Word but not LibreOffice installed — mirrors
    the win32com pattern already used by _convert_docx_to_pdf and
    _convert_legacy_doc_to_docx elsewhere in this file. Not available on
    Linux/macOS or in headless server environments without Word licensed
    and installed; LibreOffice should be preferred for production.
    """
    try:
        import importlib.util
        if not importlib.util.find_spec("win32com"):
            print("[CONTROLLER] pywin32 is unavailable; cannot use Word COM for HTML→DOCX")
            return None

        print("[CONTROLLER] Converting HTML to DOCX with Microsoft Word COM...")
        result = subprocess.run(
            [
                sys.executable, "-c",
                (
                    "import os, sys, pythoncom, win32com.client; "
                    "source, destination = sys.argv[1], sys.argv[2]; "
                    "pythoncom.CoInitialize(); word = None; doc = None; "
                    "\ntry:\n"
                    "    word = win32com.client.DispatchEx('Word.Application')\n"
                    "    word.Visible = False\n"
                    "    doc = word.Documents.Open(source, ReadOnly=True, AddToRecentFiles=False)\n"
                    "    doc.SaveAs2(destination, FileFormat=16)\n"  # 16 = wdFormatDocumentDefault (.docx)
                    "finally:\n"
                    "    if doc is not None: doc.Close(False)\n"
                    "    if word is not None: word.Quit()\n"
                    "    pythoncom.CoUninitialize()\n"
                    "sys.exit(0 if os.path.exists(destination) and os.path.getsize(destination) > 0 else 1)"
                ),
                html_path,
                docx_path,
            ],
            capture_output=True, text=True, timeout=45,
        )
        if os.path.exists(docx_path) and os.path.getsize(docx_path) > 1000:
            print(f"[CONTROLLER] HTML→DOCX via Word COM: {docx_path}")
            return docx_path
        print(f"[CONTROLLER] Word COM HTML→DOCX conversion failed; stderr: {result.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        print("[CONTROLLER] Word COM HTML→DOCX conversion timed out (45s)")
        return None
    except Exception as e:
        print(f"[CONTROLLER] Word COM HTML→DOCX unexpected error: {e}")
        return None


def _zero_out_docx_page_margins(docx_path: str) -> bool:
    """
    Zero out page margins on every section of a converted .docx.

    WHY THIS IS NEEDED: the source HTML's REQUIRED_CSS sets `@page {
    margin: 0; size: A4 portrait; }`, and Chromium (the PDF renderer)
    honors that exactly — which is why the PDF output uses the full page
    width edge-to-edge. Neither LibreOffice nor Microsoft Word's HTML
    import reliably respects CSS `@page` margin rules during HTML→DOCX
    conversion; both fall back to their own default document margins
    (commonly ~1 inch / 2.54cm on every side) regardless of what the
    source HTML specified. The table inside is still `width:100%`, but
    100% of a page area shrunk by ~2 inches of margin is visibly narrower
    than the PDF's true edge-to-edge layout — this is what makes the
    Word output look like it isn't using the full page.

    Setting every section's margins to 0 post-conversion makes the DOCX's
    usable page area match the PDF's, so the same 100%-width table now
    actually reaches the true page edges in Word too.
    """
    try:
        from docx import Document
        from docx.shared import Cm
    except ImportError:
        print("[CONTROLLER] python-docx not installed; cannot zero out DOCX page margins")
        return False

    try:
        doc = Document(docx_path)
        for section in doc.sections:
            section.top_margin = Cm(1)
            section.bottom_margin = Cm(1)
            section.left_margin = Cm(1)
            section.right_margin = Cm(1)
            # Header/footer distance also defaults to a nonzero value in
            # most conversions; zero these too so nothing reserves extra
            # space above/below the printable area.
            section.header_distance = Cm(1)
            section.footer_distance = Cm(1)
        doc.save(docx_path)
        print(f"[CONTROLLER] Zeroed out page margins on {len(doc.sections)} section(s) in {docx_path}")
        return True
    except Exception as e:
        print(f"[CONTROLLER] Failed to zero out DOCX page margins: {e}")
        return False


def _convert_html_to_docx(html_content: str, work_dir: str) -> Optional[str]:
    """
    Convert generated resume HTML to a real binary .docx.

    Tries LibreOffice headless first (works everywhere, including Linux
    production servers with no Office license needed). Falls back to
    Microsoft Word COM automation if LibreOffice isn't installed but Word
    is — useful on Windows dev/staging machines like this one, where Word
    is already relied on for legacy .doc conversion elsewhere in this file.

    Writes the HTML to a temp file inside `work_dir` (already an ephemeral
    per-request directory owned by the caller).

    Returns the path to the produced .docx file, or None if neither
    converter is available or both fail — callers must treat None as
    "fall back to the raw HTML response", never as a hard error.
    """
    work_dir = os.path.abspath(work_dir)
    html_path = os.path.join(work_dir, f"{uuid.uuid4().hex}_resume.html")

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"[CONTROLLER] Failed to write intermediate HTML for DOCX conversion: {e}")
        return None

    docx_path = os.path.splitext(html_path)[0] + ".docx"

    docx = _convert_html_to_docx_libreoffice(html_path, work_dir, docx_path)
    if not docx:
        print("[CONTROLLER] Falling back to Microsoft Word COM for HTML→DOCX...")
        docx = _convert_html_to_docx_word_com(html_path, docx_path)

    if not docx:
        print("[CONTROLLER] No available converter (LibreOffice or Word COM) could produce a DOCX.")
        return None

    # Whichever converter produced the .docx, its page margins default to
    # ~1 inch regardless of the source HTML's @page{margin:0} — fix that up
    # so the Word output actually fills the page like the PDF does. If this
    # step fails for any reason, we still return the DOCX as-is (correct
    # content, just with default margins) rather than discarding a valid
    # file over a cosmetic post-processing failure.
    _zero_out_docx_page_margins(docx)
    return docx


# ─────────────────────────────────────────────────────────────────────────────
#  Shared pipeline logic — everything happens inside one ephemeral work_dir
# ─────────────────────────────────────────────────────────────────────────────

def _convert_legacy_doc_to_docx(doc_path: str) -> Optional[str]:
    """Convert a legacy binary .doc file to .docx using Microsoft Word."""
    output_path = os.path.splitext(doc_path)[0] + ".docx"
    try:
        import importlib.util
        if not importlib.util.find_spec("win32com"):
            print("[CONTROLLER] pywin32 is unavailable; cannot convert legacy .doc")
            return None

        print("[CONTROLLER] Converting legacy .doc to .docx with Microsoft Word...")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os, sys, pythoncom, win32com.client; "
                    "source, destination = sys.argv[1], sys.argv[2]; "
                    "pythoncom.CoInitialize(); word = None; doc = None; "
                    "\ntry:\n"
                    "    word = win32com.client.DispatchEx('Word.Application')\n"
                    "    word.Visible = False\n"
                    "    doc = word.Documents.Open(source, ReadOnly=True, AddToRecentFiles=False)\n"
                    "    doc.SaveAs2(destination, FileFormat=16)\n"
                    "finally:\n"
                    "    if doc is not None: doc.Close(False)\n"
                    "    if word is not None: word.Quit()\n"
                    "    pythoncom.CoUninitialize()\n"
                    "sys.exit(0 if os.path.exists(destination) and os.path.getsize(destination) > 0 else 1)"
                ),
                doc_path,
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            print(f"[CONTROLLER] Legacy .doc converted to .docx: {output_path}")
            return output_path
        print(f"[CONTROLLER] Legacy .doc conversion failed: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("[CONTROLLER] Legacy .doc conversion timed out")
    except Exception as e:
        print(f"[CONTROLLER] Legacy .doc conversion failed: {e}")
    return None


async def _run_pipeline(
    resume_x: UploadFile,
    resume_y: UploadFile,
    work_dir: str,
    return_html: bool = False,
    llm_settings: Optional[dict] = None,
) -> dict:
    """
    Core resume formatting pipeline. All intermediate and output files are
    written inside `work_dir` — a caller-supplied ephemeral OS temp
    directory. Nothing here ever writes to a project-local folder, and
    nothing here deletes work_dir itself; the caller (format_resume /
    format_resume_html) owns that directory's full lifecycle.

    When `return_html` is True (the Word-download path), this now also
    attempts to convert the generated HTML into a real .docx via
    LibreOffice. On success, the returned dict carries `docx_base64` /
    `docx_filename` in addition to `html`; on failure it carries only
    `html`, so callers can fall back gracefully.
    """
    print(f"[TEMPLATE] Processing resume format with template: {resume_x.filename}")

    resume_x_content = await resume_x.read()
    resume_y_content = await resume_y.read()
    print(f"[CONTROLLER] X: {len(resume_x_content):,}B  Y: {len(resume_y_content):,}B")

    if not _is_valid_resume_file(resume_x, resume_x_content):
        raise HTTPException(status_code=400, detail=f"Format resume (X) must be PDF or DOCX (received: {resume_x.content_type})")
    if not _is_valid_resume_file(resume_y, resume_y_content):
        raise HTTPException(status_code=400, detail=f"Content resume (Y) must be PDF or DOCX (received: {resume_y.content_type})")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(resume_x_content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Format resume exceeds {settings.MAX_FILE_SIZE_MB} MB limit")
    if len(resume_y_content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Content resume exceeds {settings.MAX_FILE_SIZE_MB} MB limit")

    x_ext = _get_file_extension(resume_x, resume_x_content)
    y_ext = _get_file_extension(resume_y, resume_y_content)

    resume_x_path = os.path.join(work_dir, f"{uuid.uuid4().hex}_format{x_ext}")
    resume_y_path = os.path.join(work_dir, f"{uuid.uuid4().hex}_content{y_ext}")

    with open(resume_x_path, "wb") as f:
        f.write(resume_x_content)
    with open(resume_y_path, "wb") as f:
        f.write(resume_y_content)

    print(f"[CONTROLLER] Wrote to ephemeral work_dir: X={resume_x_path}  Y={resume_y_path}")

    # python-docx cannot read legacy binary .doc files. Convert them before
    # text extraction so accepted .doc uploads do not become opaque 500s.
    if x_ext == ".doc":
        converted_x = await asyncio.to_thread(_convert_legacy_doc_to_docx, resume_x_path)
        if not converted_x:
            raise HTTPException(
                status_code=400,
                detail="Could not convert the format template from legacy .doc to .docx. "
                       "Open it in Microsoft Word and save it as .docx, then upload it again.",
            )
        resume_x_path, x_ext = converted_x, ".docx"

    if y_ext == ".doc":
        converted_y = await asyncio.to_thread(_convert_legacy_doc_to_docx, resume_y_path)
        if not converted_y:
            raise HTTPException(
                status_code=400,
                detail="Could not convert the content resume from legacy .doc to .docx. "
                       "Open it in Microsoft Word and save it as .docx, then upload it again.",
            )
        resume_y_path, y_ext = converted_y, ".docx"

    resume_x_text = extract_text_from_document(resume_x_path)
    resume_y_text = extract_text_from_document(resume_y_path)
    print(f"[CONTROLLER] Text: X={len(resume_x_text)} chars  Y={len(resume_y_text)} chars")

    if not resume_x_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from format resume (Resume X). Is it a scanned image PDF? Please use a text-based PDF or DOCX.")
    if not resume_y_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from content resume (Resume Y). Is it a scanned image PDF? Please use a text-based PDF or DOCX.")

    resume_x_docx_path: Optional[str] = resume_x_path if x_ext == ".docx" else None

    resume_x_pdf_path: Optional[str] = None
    if x_ext == ".pdf":
        resume_x_pdf_path = resume_x_path
    elif x_ext in (".docx", ".doc"):
        converted = await asyncio.to_thread(_convert_docx_to_pdf, resume_x_path)
        if converted:
            resume_x_pdf_path = converted
            print(f"[CONTROLLER] Using converted PDF for vision: {converted}")
        else:
            print("[CONTROLLER] DOCX→PDF failed; Gemini will use text-only mode")

    print("[CONTROLLER] Calling Gemini pipeline...")
    html_content = generate_formatted_resume(
        resume_x_pdf_path=resume_x_pdf_path,
        resume_x_text=resume_x_text,
        resume_y_text=resume_y_text,
        resume_x_docx_path=resume_x_docx_path,
        llm_settings=llm_settings,
    )
    print(f"[CONTROLLER] HTML generated: {len(html_content):,} chars")

    result: dict = {"html": html_content, "pdf_path": None}

    if not return_html:
        print("[CONTROLLER] Generating PDF...")
        # generate_pdf_from_html uses Playwright's sync API internally, which
        # cannot run inside a thread that already has an asyncio event loop
        # active (FastAPI's request-handling thread does). Run it in a
        # separate worker thread via asyncio.to_thread so Playwright gets a
        # clean thread with no event loop — same pattern already used
        # correctly in template_render_compare.py's render_html_to_pdf_bytes.
        pdf_path = await asyncio.to_thread(generate_pdf_from_html, html_content, work_dir)
        print(f"[CONTROLLER] PDF ready: {pdf_path}")
        result["pdf_path"] = pdf_path
    else:
        # Word-download path: attempt a real .docx conversion of the exact
        # same HTML the PDF path renders. This is what makes the Word output
        # match the PDF's alignment/spacing instead of relying on Word's own
        # weak HTML filter to interpret the raw HTML.
        print("[CONTROLLER] Generating DOCX...")
        docx_path = await asyncio.to_thread(_convert_html_to_docx, html_content, work_dir)
        if docx_path:
            try:
                with open(docx_path, "rb") as f:
                    result["docx_base64"] = base64.b64encode(f.read()).decode("utf-8")
                result["docx_filename"] = "formatted_resume.docx"
                print(f"[CONTROLLER] DOCX ready: {docx_path} ({len(result['docx_base64']):,} b64 chars)")
            except Exception as e:
                print(f"[CONTROLLER] Failed to read/encode generated DOCX, falling back to HTML: {e}")
        else:
            print("[CONTROLLER] DOCX conversion unavailable/failed — falling back to raw HTML response")

    return result


def _cleanup_dir(path: str) -> None:
    """Remove an entire ephemeral work directory and everything in it."""
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"[CONTROLLER] Cleaned up work dir: {path}")
    except Exception as e:
        print(f"[CONTROLLER] Cleanup error for {path}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint handlers
# ─────────────────────────────────────────────────────────────────────────────

async def format_resume(
    resume_x: UploadFile,
    resume_y: UploadFile,
    output_format: str = "pdf",
    background_tasks: Optional[BackgroundTasks] = None,
    llm_settings: Optional[dict] = None,
) -> FileResponse:
    print(f"[CONTROLLER] format_resume called")
    print(f"[CONTROLLER] X: {resume_x.filename} ({resume_x.content_type})")
    print(f"[CONTROLLER] Y: {resume_y.filename} ({resume_y.content_type})")

    return_html = output_format.lower() in ("html", "docx")

    # Ephemeral per-request directory — lives under the OS temp location,
    # never inside the project. Always removed before this function returns
    # (JSON responses) or via BackgroundTask after streaming (PDF responses).
    work_dir = tempfile.mkdtemp(prefix="resumefmt_")

    try:
        result = await _run_pipeline(resume_x, resume_y, work_dir, return_html=return_html, llm_settings=llm_settings)

        if return_html:
            if result.get("docx_base64"):
                return JSONResponse(status_code=200, content={
                    "success": True,
                    "docx_base64": result["docx_base64"],
                    "filename": result.get("docx_filename", "formatted_resume.docx"),
                    "pdf_base64": result.get("pdf_base64"),
                    "pdf_filename": result.get("pdf_filename", "formatted_resume.pdf"),
                })
            return JSONResponse(status_code=200, content={"success": True, "html": result.get("html")})

        pdf_path = result["pdf_path"]
        if background_tasks:
            background_tasks.add_task(_cleanup_dir, work_dir)
        else:
            # No BackgroundTasks available — best effort, clean up now.
            # (FileResponse will already have streamed synchronously by then
            # in most test/dev setups; in production always pass background_tasks.)
            pass

        return FileResponse(path=pdf_path, media_type="application/pdf", filename="formatted_resume.pdf")

    except HTTPException:
        _cleanup_dir(work_dir)
        raise
    except Exception as e:
        _cleanup_dir(work_dir)
        print(f"[CONTROLLER] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # For the JSON/return_html branch we've already returned above
        # without registering a background cleanup — clean up synchronously
        # here for that case. For the FileResponse branch, cleanup is
        # already scheduled via background_tasks, so this is a no-op if the
        # dir was already removed; harmless otherwise.
        if return_html:
            _cleanup_dir(work_dir)


async def format_resume_html(
    resume_x: UploadFile,
    resume_y: UploadFile,
    llm_settings: Optional[dict] = None,
) -> JSONResponse:
    """POST /format/html — same pipeline, returns raw HTML for preview/debugging,
    or a real DOCX (base64) when LibreOffice conversion succeeds."""
    print(f"[CONTROLLER] format_resume_html called")

    work_dir = tempfile.mkdtemp(prefix="resumefmt_")

    try:
        result = await _run_pipeline(resume_x, resume_y, work_dir, return_html=True, llm_settings=llm_settings)

        if result.get("docx_base64"):
            return JSONResponse(status_code=200, content={
                "success": True,
                "html": result.get("html"),
                "docx_base64": result["docx_base64"],
                "filename": result.get("docx_filename", "formatted_resume.docx"),
                "pdf_base64": result.get("pdf_base64"),
                "pdf_filename": result.get("pdf_filename", "formatted_resume.pdf"),
            })

        return JSONResponse(status_code=200, content={"success": True, "html": result["html"]})

    except HTTPException:
        raise
    except Exception as e:
        print(f"[CONTROLLER] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        _cleanup_dir(work_dir)


async def health_check() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "resume-formatter-backend",
    }