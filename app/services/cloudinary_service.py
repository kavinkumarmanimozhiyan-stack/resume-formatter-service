import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

TEMPLATES_FOLDER_PREFIX = "resume-formatter/templates"


def upload_template(
    file_path: str,
    template_name: str,
    public_id: str,
    category: str = "default",
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload a template file to Cloudinary. `public_id` is caller-supplied
    (a uuid) so it can double as the template_id — no separate local
    mapping is needed. Display name / original filename are stored in
    Cloudinary's `context` metadata, making Cloudinary the sole source of
    truth for templates (no local JSON store).
    """
    try:
        folder = f"{TEMPLATES_FOLDER_PREFIX}/{category}"

        result = cloudinary.uploader.upload(
            file_path,
            public_id=public_id,
            folder=folder,
            resource_type="raw",
            overwrite=True,
            invalidate=True,
            context={
                "name": template_name,
                "original_filename": original_filename or "",
                "category": category,
            },
        )

        logger.info(f"[CLOUDINARY] Uploaded template: {result['public_id']}")

        return {
            "public_id": result["public_id"],
            "secure_url": result["secure_url"],
            "url": result["secure_url"],
            "resource_type": result["resource_type"],
            "format": result.get("format"),
            "bytes": result.get("bytes"),
            "created_at": result.get("created_at"),
            "version": result.get("version"),
        }
    except Exception as e:
        logger.error(f"[CLOUDINARY] Template upload failed: {str(e)}")
        raise


def download_template(public_id: str, output_path: str) -> bool:
    try:
        resource = cloudinary.api.resource(
            public_id,
            resource_type="raw",
            type="upload",
        )

        secure_url = resource.get("secure_url")

        if not secure_url:
            logger.error(
                f"[CLOUDINARY] No secure_url returned for {public_id}"
            )
            return False

        logger.info(
            f"[CLOUDINARY] Resource found: {resource.get('public_id')}"
        )

        logger.info(
            f"[CLOUDINARY] secure_url: {secure_url}"
        )

        response = requests.get(
            secure_url,
            timeout=30,
        )

        response.raise_for_status()

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(
            f"[CLOUDINARY] Template downloaded successfully: {output_path}"
        )

        return True

    except Exception as e:
        logger.error(
            f"[CLOUDINARY] Template download failed: {e}"
        )
        return False
    
def _context_dict(resource: Dict[str, Any]) -> Dict[str, str]:
    """Cloudinary returns context as {'custom': {...}} — normalize it."""
    ctx = resource.get("context") or {}
    return ctx.get("custom", ctx) if isinstance(ctx, dict) else {}


def _resource_to_template_meta(resource: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _context_dict(resource)
    public_id = resource["public_id"]
    template_id = public_id.rsplit("/", 1)[-1]
    display_name = ctx.get("name") or template_id
    return {
        "template_id": template_id,
        # Short aliases the frontend uses (`t.id`, `t.name`) — without these,
        # selects/buttons key off undefined values and delete/rename break.
        "id": template_id,
        "name": display_name,
        "template_name": display_name,
        "original_filename": ctx.get("original_filename") or "",
        "cloudinary_public_id": public_id,
        "cloudinary_url": resource.get("secure_url"),
        "resource_type": resource.get("resource_type", "raw"),
        "format": resource.get("format"),
        "size_bytes": resource.get("bytes"),
        "category": ctx.get("category") or "default",
        "created_at": resource.get("created_at"),
    }


def list_templates(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all templates directly from Cloudinary — no local metadata file."""
    try:
        prefix = f"{TEMPLATES_FOLDER_PREFIX}/{category}/" if category else f"{TEMPLATES_FOLDER_PREFIX}/"
        result = cloudinary.api.resources(
            type="upload",
            resource_type="raw",
            prefix=prefix,
            context=True,
            max_results=500,
        )
        items = [_resource_to_template_meta(r) for r in result.get("resources", [])]
        items.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        logger.info(f"[CLOUDINARY] Listed {len(items)} templates")
        return items
    except Exception as e:
        logger.error(f"[CLOUDINARY] Failed to list templates: {str(e)}")
        return []


def find_template_public_id(template_id: str) -> Optional[str]:
    """
    template_id is just the tail segment of the public_id (the uuid we
    chose at upload time). We don't know its category/folder up front, so
    search for it among all templates. Cheap enough at template-catalog
    scale; swap for a direct api.resource() call if you later encode the
    category into template_id itself.
    """
    for meta in list_templates():
        if meta["template_id"] == template_id:
            return meta["cloudinary_public_id"]
    return None


def get_template_metadata(template_id: str) -> Optional[Dict[str, Any]]:
    """Get a single template's metadata straight from Cloudinary."""
    public_id = find_template_public_id(template_id)
    if not public_id:
        return None
    try:
        resource = cloudinary.api.resource(public_id, resource_type="raw", type="upload", context=True)
        return _resource_to_template_meta(resource)
    except Exception as e:
        logger.error(f"[CLOUDINARY] Failed to get template metadata for {template_id}: {str(e)}")
        return None


def rename_template(template_id: str, new_name: str) -> Optional[Dict[str, Any]]:
    """Update a template's display name via Cloudinary context metadata."""
    public_id = find_template_public_id(template_id)
    if not public_id:
        return None
    try:
        cloudinary.api.update(
            public_id,
            resource_type="raw",
            type="upload",
            context=f"name={new_name}",
        )
        return get_template_metadata(template_id)
    except Exception as e:
        logger.error(f"[CLOUDINARY] Failed to rename template {template_id}: {str(e)}")
        return None


def delete_template(template_id: str) -> bool:
    """Delete a template from Cloudinary by template_id."""
    public_id = find_template_public_id(template_id)
    if not public_id:
        return False
    try:
        result = cloudinary.uploader.destroy(public_id, resource_type="raw", invalidate=True)
        logger.info(f"[CLOUDINARY] Deleted template: {public_id}")
        return result.get("result") == "ok"
    except Exception as e:
        logger.error(f"[CLOUDINARY] Failed to delete template {template_id}: {str(e)}")
        return False