#!/usr/bin/env python3
"""
Migration utility to upload existing local templates to Cloudinary.

This script reads templates from the local templates/ folder and uploads
them to Cloudinary, storing metadata so the application can use them.

Usage:
    python migrate_templates_to_cloudinary.py
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to path so we can import app modules
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(backend_dir, ".."))

from app.services.cloudinary_service import upload_template
from app.services.cloudinary_template_store import save_template_metadata
from app.init_cloudinary import init_cloudinary

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def migrate_templates():
    """Migrate existing local templates to Cloudinary"""
    
    # Initialize Cloudinary
    logger.info("Initializing Cloudinary...")
    if not init_cloudinary():
        logger.error("Failed to initialize Cloudinary. Check CLOUDINARY_* environment variables.")
        return False
    
    templates_dir = os.path.join(backend_dir, "..", "templates")
    if not os.path.exists(templates_dir):
        logger.info(f"No templates directory found at {templates_dir}")
        return True
    
    # Find all template files (not .json metadata files)
    template_files = []
    for item in os.listdir(templates_dir):
        item_path = os.path.join(templates_dir, item)
        if os.path.isfile(item_path) and not item.endswith(".json"):
            template_files.append((item, item_path))
    
    if not template_files:
        logger.info("No template files found to migrate")
        return True
    
    logger.info(f"Found {len(template_files)} template files to migrate")
    
    successful = 0
    failed = 0
    
    for filename, file_path in template_files:
        try:
            logger.info(f"Uploading: {filename}")
            
            # Get file extension
            _, ext = os.path.splitext(filename)
            ext = ext.lstrip(".")
            
            # Generate template name from filename
            template_name = Path(filename).stem.replace("_", " ").title()
            
            # Upload to Cloudinary
            result = upload_template(
                file_path=file_path,
                template_name=template_name,
                category="default",
            )
            
            # Save metadata
            save_template_metadata(
                template_name=template_name,
                original_filename=filename,
                cloudinary_public_id=result["public_id"],
                cloudinary_url=result["secure_url"],
                resource_type=result.get("resource_type", "raw"),
                format=ext,
                size_bytes=os.path.getsize(file_path),
                category="default",
            )
            
            logger.info(f"✓ Uploaded: {filename}")
            successful += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to upload {filename}: {str(e)}")
            failed += 1
    
    logger.info(f"\nMigration complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    
    if failed == 0:
        logger.info("\nAll templates migrated successfully!")
        logger.info("You can now safely remove or rename the local templates/ folder.")
    
    return failed == 0


if __name__ == "__main__":
    success = migrate_templates()
    sys.exit(0 if success else 1)
