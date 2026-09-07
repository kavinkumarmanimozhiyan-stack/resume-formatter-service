"""
template_store.py

Templates are stored ONLY in Cloudinary — no local metadata file, no local
templates/ directory. Cloudinary's `context` metadata carries the display
name; template_id is the tail segment of the Cloudinary public_id.

Resume uploads (format/content files) are handled separately and are never
persisted here or anywhere on local disk — see resume_controller.py, which
uses a per-request temp directory that is deleted immediately after use.
"""

import logging
import re
import tempfile
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^a-zA-Z0-9 _\-().]", "", name)
    return name[:80] if name else "Untitled Template"


def _detect_ext(filename: Optional[str], content_type: Optional[str], content: bytes) -> str:
    if content[:4] == b"%PDF":
        return ".pdf"
    if content[:2] == b"PK":
        return ".docx"

    if filename:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return ".pdf"
        if lower.endswith(".docx"):
            return ".docx"
        if lower.endswith(".doc"):
            return ".doc"

    ct = (content_type or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "wordprocessingml" in ct or "msword" in ct:
        return ".docx"

    return ".pdf"


def save_template_to_cloudinary(
    *,
    file_bytes: bytes,
    original_filename: Optional[str],
    content_type: Optional[str],
    name: Optional[str],
    category: str = "default",
) -> Dict[str, Any]:
    """
    Upload a template to Cloudinary. The uploaded bytes touch local disk
    only as a short-lived OS temp file for the duration of the upload call
    itself, then are removed immediately in the `finally` block.
    """
    from app.services.cloudinary_service import upload_template

    ext = _detect_ext(original_filename, content_type, file_bytes)
    template_display_name = _safe_name(name or (original_filename or "").rsplit(".", 1)[0])
    template_id = uuid.uuid4().hex

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            temp_file = tmp.name

        cloudinary_result = upload_template(
            file_path=temp_file,
            template_name=template_display_name,
            public_id=template_id,
            category=category,
            original_filename=original_filename,
        )

        metadata = {
            "template_id": template_id,
            "template_name": template_display_name,
            "original_filename": original_filename or "template",
            "cloudinary_public_id": cloudinary_result["public_id"],
            "cloudinary_url": cloudinary_result["secure_url"],
            "resource_type": cloudinary_result.get("resource_type", "raw"),
            "format": ext.lstrip("."),
            "size_bytes": len(file_bytes),
            "category": category,
        }
        logger.info(f"[TEMPLATE_STORE] Template uploaded to Cloudinary: {template_id}")
        return metadata

    except Exception as e:
        logger.error(f"[TEMPLATE_STORE] Failed to upload template to Cloudinary: {str(e)}")
        raise
    finally:
        if temp_file:
            import os
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"[TEMPLATE_STORE] Failed to delete temp upload file: {str(e)}")


def download_template_from_cloudinary(template_id: str) -> Optional[str]:
    """
    Download a template to a short-lived OS temp file. Caller is
    responsible for deleting this path once done with it (resume_controller
    already does this via its per-request temp-dir cleanup).
    """
    from app.services.cloudinary_service import download_template, get_template_metadata

    meta = get_template_metadata(template_id)
    if not meta:
        logger.warning(f"[TEMPLATE_STORE] Template metadata not found: {template_id}")
        return None

    public_id = meta.get("cloudinary_public_id")
    if not public_id:
        logger.warning(f"[TEMPLATE_STORE] No Cloudinary public_id for template: {template_id}")
        return None

    # Cloudinary `raw` resources often have no `format` key — infer from the
    # original filename / public_id, defaulting to docx only as a last resort.
    ext = meta.get("format") or ""
    if not ext:
        for candidate in (meta.get("original_filename"), meta.get("cloudinary_public_id")):
            if candidate and "." in candidate:
                ext = candidate.rsplit(".", 1)[1].lower()
                break
    ext = ext or "docx"
    if not ext.startswith("."):
        ext = f".{ext}"

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name

    success = download_template(public_id, temp_file)
    if success:
        logger.info(f"[TEMPLATE_STORE] Downloaded template from Cloudinary: {template_id}")
        return temp_file

    logger.error(f"[TEMPLATE_STORE] Failed to download template from Cloudinary: {template_id}")
    return None


def list_templates() -> List[Dict[str, Any]]:
    from app.services.cloudinary_service import list_templates as cloudinary_list_templates
    return cloudinary_list_templates()


def get_template_meta(template_id: str) -> Optional[Dict[str, Any]]:
    from app.services.cloudinary_service import get_template_metadata
    return get_template_metadata(template_id)


def get_template_file_path(template_id: str) -> Optional[str]:
    """
    Download a template to a temp path for one-time use by the Gemini
    pipeline. This path is NOT tracked or cleaned up here — callers must
    delete it themselves once they're finished (resume_controller's
    per-request temp-dir cleanup handles this).
    """
    return download_template_from_cloudinary(template_id)


def delete_template(template_id: str) -> bool:
    from app.services.cloudinary_service import delete_template as cloudinary_delete_template
    ok = cloudinary_delete_template(template_id)
    logger.info(f"[TEMPLATE_STORE] Deleted template: {template_id} (success={ok})")
    return ok


def rename_template(template_id: str, new_name: str) -> Optional[Dict[str, Any]]:
    from app.services.cloudinary_service import rename_template as cloudinary_rename_template
    return cloudinary_rename_template(template_id, _safe_name(new_name))