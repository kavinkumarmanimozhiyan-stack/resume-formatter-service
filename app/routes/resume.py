from fastapi import APIRouter, Request, BackgroundTasks, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from tempfile import SpooledTemporaryFile
import os
import time
from typing import Optional

from app.controllers.resume_controller import format_resume, health_check
from app.services.template_store import (
    list_templates,
    get_template_file_path,
    delete_template,
    get_template_meta,
    rename_template as rename_template_meta,
)
from app.services.llm_client import generate_llm_text, get_llm_runtime
from app.services.llm_settings_store import (
    delete_llm_settings,
    get_llm_setting_by_id,
    get_llm_settings,
    list_llm_settings,
    resolve_llm_settings,
    save_llm_settings,
)

router = APIRouter()


def _llm_settings_from_form(form) -> dict:
    return resolve_llm_settings({
        "id": str(form.get("llm_settings_id") or form.get("llmSettingsId") or "").strip(),
        "provider": str(form.get("llm_provider") or form.get("provider") or "gemini").strip().lower(),
        "model": str(form.get("llm_model") or form.get("model") or "").strip(),
        "api_key": str(form.get("llm_api_key") or form.get("api_key") or "").strip(),
    })


@router.get("/llm-settings")
async def read_llm_settings():
    stored = get_llm_settings(include_api_key=False)
    return {"success": True, "settings": stored, "saved_settings": list_llm_settings(include_api_key=False)}


@router.get("/llm-settings/{setting_id}")
async def read_llm_setting(setting_id: str):
    stored = get_llm_setting_by_id(setting_id, include_api_key=True)
    if not stored:
        raise HTTPException(status_code=404, detail="Stored LLM setting not found.")
    return {"success": True, "settings": stored}


@router.put("/llm-settings")
async def update_llm_settings(payload: dict):
    try:
        stored = save_llm_settings(
            setting_id=str((payload or {}).get("id") or (payload or {}).get("settingsId") or "").strip(),
            provider=str((payload or {}).get("provider") or "").strip(),
            model=str((payload or {}).get("model") or "").strip(),
            api_key=str((payload or {}).get("apiKey") or (payload or {}).get("api_key") or "").strip(),
            label=str((payload or {}).get("label") or "").strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "settings": stored}


@router.delete("/llm-settings")
async def remove_llm_settings():
    delete_llm_settings()
    return {"success": True}


@router.post("/llm-settings/test")
async def test_llm_settings(payload: dict):
    draft = {
        "provider": str((payload or {}).get("provider") or "").strip().lower(),
        "model": str((payload or {}).get("model") or "").strip(),
        "api_key": str((payload or {}).get("apiKey") or (payload or {}).get("api_key") or "").strip(),
        "id": str((payload or {}).get("id") or (payload or {}).get("settingsId") or "").strip(),
    }
    runtime_settings = resolve_llm_settings(draft)
    try:
        client, provider, active_model, api_key = get_llm_runtime(runtime_settings)
        started_at = time.perf_counter()
        text = generate_llm_text(
            provider=provider,
            client=client,
            model=active_model,
            api_key=api_key,
            prompt="Reply with exactly: connection ok",
            system_instruction="You are a connection test endpoint.",
            temperature=0.0,
            max_output_tokens=32,
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="Provider returned an empty response.")

    return {
        "success": True,
        "usage": {
            "latencyMs": latency_ms,
            "estimatedOutputChars": len(text),
            "lastTestedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


@router.get("/templates")
async def get_templates():
    return {"success": True, "templates": list_templates()}


@router.post("/templates")
async def create_template(request: Request):
    """
    Upload a resume template — goes to Cloudinary ONLY (no local disk
    persistence). This is the one place a user explicitly asks to save
    something long-term.
    """
    from app.services.template_store import save_template_to_cloudinary

    form = await request.form()
    up = form.get("file") or form.get("template") or form.get("format_template") or form.get("resume_x")
    name = form.get("name") or form.get("template_name")

    if not up or not hasattr(up, "filename") or not up.filename:
        raise HTTPException(status_code=400, detail="Missing template file field 'file'")

    file_bytes = await up.read()

    try:
        meta = save_template_to_cloudinary(
            file_bytes=file_bytes,
            original_filename=getattr(up, "filename", None),
            content_type=getattr(up, "content_type", None),
            name=str(name) if name else None,
        )
        return {"success": True, "template": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload template: {str(e)}")


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    meta = get_template_meta(template_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True, "template": meta}


@router.delete("/templates/{template_id}")
async def remove_template(template_id: str):
    ok = delete_template(template_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True}


@router.patch("/templates/{template_id}")
async def rename_template(template_id: str, payload: dict):
    new_name = (payload or {}).get("name")
    if not new_name or not str(new_name).strip():
        raise HTTPException(status_code=400, detail="Missing 'name' in request body")

    meta = rename_template_meta(template_id, str(new_name))
    if not meta:
        raise HTTPException(status_code=404, detail="Template not found")

    return {"success": True, "template": meta}


def _uploadfile_from_path(path: str, filename: Optional[str] = None, delete_after_read: bool = False) -> UploadFile:
    """
    Create an UploadFile-like object backed by an in-memory spooled file,
    reading `path` once. If `delete_after_read` is True (used for templates
    downloaded from Cloudinary into a temp file), the source file on disk is
    removed immediately after being read into memory — it's never touched
    again, so there's no reason to let it linger.
    """
    stf = SpooledTemporaryFile(max_size=10 * 1024 * 1024)
    with open(path, "rb") as f:
        stf.write(f.read())
    stf.seek(0)

    if delete_after_read:
        try:
            os.remove(path)
        except Exception as e:
            print(f"[ROUTE] Warning: failed to remove downloaded template temp file {path}: {e}")

    return StarletteUploadFile(file=stf, filename=filename or os.path.basename(path))


@router.post("/generate-resume")
async def generate_resume_route(request: Request, background_tasks: BackgroundTasks = None):
    form = await request.form()

    print(f"[ROUTE] Incoming form fields: {list(form.keys())}")
    for key, value in form.items():
        display_value = "[REDACTED]" if "key" in key.lower() else getattr(value, "filename", value)
        print(f"[ROUTE] Field '{key}': type={type(value).__name__}, filename={display_value}")

    output_format = form.get("output_format", "html")
    template_id = form.get("template_id") or form.get("templateId") or form.get("template")

    FORMAT_KEYS = ["format_template", "resume_x", "resumeX", "formatResume", "formatFile", "format_file", "template", "file1", "format"]
    CONTENT_KEYS = ["content_file", "resume_y", "resumeY", "contentResume", "contentFile", "content_file", "resume", "file2", "content"]

    format_file = None
    content_file = None

    for key in FORMAT_KEYS:
        if key in form and hasattr(form[key], "filename") and form[key].filename:
            format_file = form[key]
            print(f"[ROUTE] Found format file under key: '{key}' -> {format_file.filename}")
            break

    for key in CONTENT_KEYS:
        if key in form and hasattr(form[key], "filename") and form[key].filename:
            content_file = form[key]
            print(f"[ROUTE] Found content file under key: '{key}' -> {content_file.filename}")
            break

    # Selecting a previously saved (Cloudinary) template instead of a fresh upload
    if not format_file and template_id:
        meta = get_template_meta(str(template_id))
        if not meta:
            return JSONResponse(status_code=404, content={"success": False, "error": "Template not found"})

        print(f"[ROUTE] Template {template_id} is in Cloudinary, downloading from: {meta.get('cloudinary_public_id')}")

        template_path = get_template_file_path(str(template_id))
        if not template_path:
            return JSONResponse(status_code=404, content={"success": False, "error": "Failed to retrieve template"})
        # Read into memory then delete the downloaded temp file immediately —
        # nothing from Cloudinary lingers on disk beyond this one read.
        format_file = _uploadfile_from_path(template_path, delete_after_read=True)
        print(f"[ROUTE] Using template_id={template_id} -> {format_file.filename}")

    if not format_file or not content_file:
        all_files = [(k, v) for k, v in form.items() if hasattr(v, "filename") and v.filename]
        print(f"[ROUTE] Positional fallback — all uploaded files: {[(k, v.filename) for k, v in all_files]}")

        if len(all_files) >= 2:
            format_file = all_files[0][1]
            content_file = all_files[1][1]
            print(f"[ROUTE] Using positional: format={format_file.filename}, content={content_file.filename}")
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Could not find both files. Received fields: {list(form.keys())}. "
                             f"Files found: {[(k, v.filename) for k, v in all_files]}"
                }
            )

    return await format_resume(format_file, content_file, output_format, background_tasks, _llm_settings_from_form(form))


@router.post("/format")
async def format_resume_route(request: Request, background_tasks: BackgroundTasks = None):
    form = await request.form()
    format_file = form.get("resume_x")
    content_file = form.get("resume_y")
    output_format = form.get("output_format", "html")

    if not format_file or not content_file:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Both resume_x and resume_y are required"}
        )

    return await format_resume(format_file, content_file, output_format, background_tasks, _llm_settings_from_form(form))


@router.post("/generate-resume-html")
async def generate_resume_html_route(request: Request):
    """Returns HTML instead of PDF for Word download"""
    from app.controllers.resume_controller import format_resume_html

    form = await request.form()
    template_id = form.get("template_id") or form.get("templateId") or form.get("template")

    FORMAT_KEYS = ["format_template", "resume_x", "resumeX", "formatResume", "formatFile", "format_file", "template", "file1", "format"]
    CONTENT_KEYS = ["content_file", "resume_y", "resumeY", "contentResume", "contentFile", "content_file", "resume", "file2", "content"]

    format_file = None
    content_file = None

    for key in FORMAT_KEYS:
        if key in form and hasattr(form[key], "filename") and form[key].filename:
            format_file = form[key]
            break

    for key in CONTENT_KEYS:
        if key in form and hasattr(form[key], "filename") and form[key].filename:
            content_file = form[key]
            break

    if not format_file and template_id:
        meta = get_template_meta(str(template_id))
        if not meta:
            return JSONResponse(status_code=404, content={"success": False, "error": "Template not found"})

        print(f"[ROUTE] Template {template_id} is in Cloudinary, downloading from: {meta.get('cloudinary_public_id')}")

        template_path = get_template_file_path(str(template_id))
        if not template_path:
            return JSONResponse(status_code=404, content={"success": False, "error": "Failed to retrieve template"})
        format_file = _uploadfile_from_path(template_path, delete_after_read=True)

    if not format_file or not content_file:
        all_files = [(k, v) for k, v in form.items() if hasattr(v, "filename") and v.filename]
        if len(all_files) >= 2:
            format_file = all_files[0][1]
            content_file = all_files[1][1]
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Could not find both files"}
            )

    return await format_resume_html(format_file, content_file, _llm_settings_from_form(form))


@router.get("/health")
async def health_check_route():
    return await health_check()