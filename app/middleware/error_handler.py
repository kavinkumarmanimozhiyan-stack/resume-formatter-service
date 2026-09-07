from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse


async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for all unhandled exceptions.
    Returns a standardized JSON error response.
    """
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)}
    )
