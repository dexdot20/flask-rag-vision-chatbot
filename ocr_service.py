from __future__ import annotations

import os
import threading
from io import BytesIO

from config import (
    OCR_DISABLED_FEATURE_ERROR,
    OCR_ENABLED,
    OCR_PRELOAD_ON_STARTUP,
    OCR_PROVIDER,
    OCR_SUPPORTED_PROVIDERS,
)
from vision import optimize_image_for_processing

_ocr_engine = None
_ocr_engine_lock = threading.Lock()


def _resolve_provider_name() -> str:
    provider = str(OCR_PROVIDER or "").strip().lower()
    if provider not in OCR_SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(OCR_SUPPORTED_PROVIDERS))
        raise RuntimeError(f"Unsupported OCR provider: {provider or 'empty'}. Supported providers: {supported}.")
    return provider


def _easyocr_gpu_enabled() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _build_easyocr_engine() -> dict:
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR dependencies are missing. Ensure easyocr and its torch dependencies are installed."
        ) from exc

    reader = easyocr.Reader(["en"], gpu=_easyocr_gpu_enabled())
    return {
        "provider": "easyocr",
        "reader": reader,
    }


def _build_paddleocr_engine() -> dict:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR dependencies are missing. Ensure paddleocr and a compatible paddlepaddle runtime are installed."
        ) from exc

    reader = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    return {
        "provider": "paddleocr",
        "reader": reader,
    }


def _build_ocr_engine(provider: str) -> dict:
    if provider == "easyocr":
        return _build_easyocr_engine()
    if provider == "paddleocr":
        return _build_paddleocr_engine()

    supported = ", ".join(sorted(OCR_SUPPORTED_PROVIDERS))
    raise RuntimeError(f"Unsupported OCR provider: {provider or 'empty'}. Supported providers: {supported}.")


def _iter_provider_candidates(preferred: str):
    yield preferred
    for provider in sorted(OCR_SUPPORTED_PROVIDERS):
        if provider != preferred:
            yield provider


def _is_missing_dependency_error(exc: Exception) -> bool:
    if isinstance(exc, ImportError):
        return True
    if isinstance(exc.__cause__, ImportError):
        return True
    message = str(exc).strip().lower()
    return "dependencies are missing" in message


def get_ocr_engine() -> dict:
    global _ocr_engine

    if not OCR_ENABLED:
        raise RuntimeError(OCR_DISABLED_FEATURE_ERROR)

    provider = _resolve_provider_name()
    if _ocr_engine is not None and _ocr_engine.get("configured_provider") == provider:
        return _ocr_engine

    with _ocr_engine_lock:
        if _ocr_engine is not None and _ocr_engine.get("configured_provider") == provider:
            return _ocr_engine

        last_error = None
        for candidate_provider in _iter_provider_candidates(provider):
            try:
                engine = _build_ocr_engine(candidate_provider)
            except RuntimeError as exc:
                if _is_missing_dependency_error(exc):
                    last_error = exc
                    continue
                raise

            engine["configured_provider"] = provider
            _ocr_engine = engine
            return _ocr_engine

        supported = ", ".join(sorted(OCR_SUPPORTED_PROVIDERS))
        message = (
            f"OCR provider '{provider}' could not be loaded because its dependencies are missing. "
            f"Install the matching OCR stack ({supported}) or set OCR_ENABLED=false."
        )
        raise RuntimeError(message) from last_error


def preload_ocr_engine(app) -> None:
    if not OCR_ENABLED or not OCR_PRELOAD_ON_STARTUP:
        return

    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if app.debug and not is_reloader_child:
        print("[startup] Flask reloader main process: OCR preload skipped.")
        return

    provider = _resolve_provider_name()
    print(f"[startup] Loading OCR engine ({provider})...")
    try:
        engine = get_ocr_engine()
    except RuntimeError as exc:
        if not _is_missing_dependency_error(exc):
            raise
        print(f"[startup] OCR preload skipped: {exc}")
        return

    resolved_provider = engine.get("provider", provider)
    if resolved_provider != provider:
        print(f"[startup] OCR provider '{provider}' unavailable; using '{resolved_provider}' instead.")
    print(f"[startup] OCR engine ready ({resolved_provider}).")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _coerce_mapping(value):
    if isinstance(value, dict):
        return value

    keys = getattr(value, "keys", None)
    if callable(keys):
        try:
            return {key: value[key] for key in value.keys()}
        except Exception:
            pass

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    raw_dict = getattr(value, "__dict__", None)
    if isinstance(raw_dict, dict):
        return raw_dict

    return None


def _extract_paddle_text_lines(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            lines.extend(_extract_paddle_text_lines(item))
        return _dedupe_preserve_order(lines)

    mapping = _coerce_mapping(value)
    if mapping is None:
        return []

    direct_texts = mapping.get("rec_texts")
    if isinstance(direct_texts, list):
        return _dedupe_preserve_order([str(item or "").strip() for item in direct_texts])

    single_text = str(mapping.get("rec_text") or mapping.get("text") or "").strip()
    if single_text:
        return [single_text]

    nested_lines = []
    for key in ("res", "result", "prunedResult", "ocrResults"):
        nested_lines.extend(_extract_paddle_text_lines(mapping.get(key)))
    return _dedupe_preserve_order(nested_lines)


def _run_easyocr(image_bytes: bytes) -> str:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow and numpy are required for EasyOCR image processing.") from exc

    engine = get_ocr_engine()
    reader = engine["reader"]
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image_array = np.array(image)

    result = reader.readtext(image_array, detail=0, paragraph=False)
    lines = _dedupe_preserve_order([str(item or "").strip() for item in result if str(item or "").strip()])
    return "\n".join(lines).strip()


def _run_paddleocr(image_bytes: bytes) -> str:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow and numpy are required for PaddleOCR image processing.") from exc

    engine = get_ocr_engine()
    reader = engine["reader"]
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image_array = np.array(image)

    result = reader.predict(
        input=image_array,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    lines = _extract_paddle_text_lines(result)
    return "\n".join(lines).strip()


def extract_image_text(image_bytes: bytes, mime_type: str) -> str:
    if not OCR_ENABLED:
        raise RuntimeError(OCR_DISABLED_FEATURE_ERROR)

    optimized_bytes, _ = optimize_image_for_processing(image_bytes, mime_type)
    provider = _resolve_provider_name()
    if provider == "easyocr":
        return _run_easyocr(optimized_bytes)
    return _run_paddleocr(optimized_bytes)