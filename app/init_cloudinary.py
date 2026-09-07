"""
Cloudinary initialization module.

This module initializes Cloudinary with credentials from environment variables.
Should be called at application startup.
"""

import cloudinary
import cloudinary.api
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def init_cloudinary():
    """Initialize Cloudinary with credentials from environment."""
    if not settings.CLOUDINARY_CLOUD_NAME:
        logger.warning("[CLOUDINARY] CLOUDINARY_CLOUD_NAME not set in environment")
        return False

    if not settings.CLOUDINARY_API_KEY:
        logger.warning("[CLOUDINARY] CLOUDINARY_API_KEY not set in environment")
        return False

    if not settings.CLOUDINARY_API_SECRET:
        logger.warning("[CLOUDINARY] CLOUDINARY_API_SECRET not set in environment")
        return False

    try:
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True,
        )

        cloudinary.api.ping()
        logger.info(f"[CLOUDINARY] Initialized with cloud_name: {settings.CLOUDINARY_CLOUD_NAME}")
        logger.info("[CLOUDINARY] Credential validation succeeded against Cloudinary API")
        return True
    except Exception as e:
        logger.error(
            "[CLOUDINARY] Cloudinary credentials rejected by the API. "
            "Check that CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET are all for the same Cloudinary account."
        )
        logger.error(f"[CLOUDINARY] Failed to initialize Cloudinary: {str(e)}")
        return False
