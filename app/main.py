import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.resume import router as resume_router
from app.middleware.error_handler import global_exception_handler
from app.services.llm_settings_store import init_llm_settings_store
from app.init_cloudinary import init_cloudinary


@asynccontextmanager
async def lifespan(app: FastAPI):
    # NOTE: No local "uploads/", "output/", or "templates/" folders are
    # created anymore. Resume files are processed entirely in ephemeral OS
    # temp directories (see resume_controller.py) that are deleted
    # immediately after each request. Templates the user explicitly saves
    # go to Cloudinary only (see template_store.py) — nothing local.
    #
    # "data/" is kept only for the LLM settings store (small app config,
    # not resume/template content). Remove this too if you'd rather keep
    # LLM settings purely in-memory or in env vars.
    os.makedirs("data", exist_ok=True)

    cloudinary_ready = init_cloudinary()
    if cloudinary_ready:
        print("Cloudinary initialized successfully")
    else:
        print("WARNING: Cloudinary not fully configured - templates cannot be saved")

    init_llm_settings_store()
    print("Application startup complete")
    yield
    print("Application shutdown")


app = FastAPI(
    title="Resume Formatter Backend",
    description="API for formatting resumes using Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# Strip whitespace around each origin so entries like
# "http://localhost:3000, http://localhost:3001" (space after the comma)
# don't silently fail to match.
allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
print(f"[STARTUP] ALLOWED_ORIGINS raw setting: {settings.ALLOWED_ORIGINS!r}")
print(f"[STARTUP] CORS allowed origins (parsed): {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    return response


# `/api` is the documented integration base for independent frontend apps.
# The unprefixed routes remain available for the bundled legacy frontend.
app.include_router(resume_router, prefix="/api", tags=["resume"])
app.include_router(resume_router, tags=["legacy"], include_in_schema=False)
app.include_router(resume_router, prefix="/api/resume", tags=["legacy"], include_in_schema=False)


@app.get("/")
async def root():
    return {
        "message": "Resume Formatter Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /api/health",
            "generate_resume": "POST /api/generate-resume",
            "api_docs": "GET /docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
