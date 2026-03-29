from __future__ import annotations

from config import IMAGE_UPLOADS_DISABLED_FEATURE_ERROR, IMAGE_UPLOADS_ENABLED, OCR_ENABLED, VISION_ENABLED
from ocr_service import extract_image_text
from vision import normalize_image_analysis, run_image_vision_analysis


def analyze_uploaded_image(image_bytes: bytes, mime_type: str, user_text: str = "") -> dict:
    if not IMAGE_UPLOADS_ENABLED:
        raise RuntimeError(IMAGE_UPLOADS_DISABLED_FEATURE_ERROR)

    combined_analysis: dict = {}
    if OCR_ENABLED:
        combined_analysis["ocr_text"] = extract_image_text(image_bytes, mime_type)

    if VISION_ENABLED:
        combined_analysis.update(run_image_vision_analysis(image_bytes, mime_type, user_text=user_text))

    return normalize_image_analysis(combined_analysis)