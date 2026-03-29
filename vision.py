from __future__ import annotations

import base64
import json
import os
import re
import threading
from io import BytesIO

from config import (
    IMAGE_ALLOWED_MIME_TYPES,
    IMAGE_MAX_BYTES,
    VISION_ATTENTION_IMPL,
    VISION_LOAD_IN_4BIT,
    VISION_MAX_IMAGE_SIDE,
    VISION_MAX_NEW_TOKENS,
    VISION_MAX_PIXELS,
    VISION_MIN_PIXELS,
    VISION_MODEL_PATH,
    VISION_PRELOAD_ON_STARTUP,
    VISION_TORCH_DTYPE_NAME,
    VISION_DISABLED_FEATURE_ERROR,
    VISION_ENABLED,
)

_vision_engine = None
_vision_engine_lock = threading.Lock()
_vision_inference_lock = threading.Lock()


def get_local_vision_engine() -> dict:
    global _vision_engine

    if not VISION_ENABLED:
        raise RuntimeError(VISION_DISABLED_FEATURE_ERROR)

    if _vision_engine is not None:
        return _vision_engine

    with _vision_engine_lock:
        if _vision_engine is not None:
            return _vision_engine

        if not VISION_MODEL_PATH:
            raise RuntimeError("Local vision model path is missing. QWEN_VL_MODEL_PATH must be set.")
        if not os.path.isdir(VISION_MODEL_PATH):
            raise RuntimeError(f"Local vision model folder not found: {VISION_MODEL_PATH}")

        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen vision dependencies are missing. Ensure torch, torchvision, transformers, qwen-vl-utils and Pillow are installed."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local Qwen vision model. No CUDA-capable GPU was detected.")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(VISION_TORCH_DTYPE_NAME, torch.float16)

        def build_model_kwargs(*, use_4bit: bool) -> dict:
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": {"": torch.cuda.current_device()},
                "local_files_only": True,
                "low_cpu_mem_usage": True,
            }
            if VISION_ATTENTION_IMPL:
                model_kwargs["attn_implementation"] = VISION_ATTENTION_IMPL

            if use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise RuntimeError(
                        "4-bit Qwen loading requires bitsandbytes-supported transformers installation."
                    ) from exc

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            return model_kwargs

        model_kwargs = build_model_kwargs(use_4bit=VISION_LOAD_IN_4BIT)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VISION_MODEL_PATH,
            **model_kwargs,
        )

        processor = AutoProcessor.from_pretrained(
            VISION_MODEL_PATH,
            local_files_only=True,
            min_pixels=VISION_MIN_PIXELS,
            max_pixels=VISION_MAX_PIXELS,
            use_fast=False,
        )
        model.eval()
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None

        _vision_engine = {
            "torch": torch,
            "model": model,
            "processor": processor,
            "process_vision_info": process_vision_info,
            "torch_dtype": torch_dtype,
        }
        return _vision_engine


def preload_local_vision_engine(app) -> None:
    if not VISION_ENABLED or not VISION_MODEL_PATH or not VISION_PRELOAD_ON_STARTUP:
        return

    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if app.debug and not is_reloader_child:
        print("[startup] Flask reloader main process: Qwen preload skipped.")
        return

    print("[startup] Loading local Qwen vision model...")
    get_local_vision_engine()
    print("[startup] Local Qwen vision model ready.")


def extract_text_from_response_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return ""


def extract_json_object(raw_text: str) -> dict:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end > start:
        candidate = raw_text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def normalize_analysis_list(values, limit: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text[:300])
    return normalized[:limit]


def normalize_image_analysis(raw_analysis: dict, fallback_text: str = "") -> dict:
    raw_analysis = raw_analysis if isinstance(raw_analysis, dict) else {}
    raw_summary = str(raw_analysis.get("vision_summary") or "").strip()
    normalized = {
        "ocr_text": str(raw_analysis.get("ocr_text") or "").strip(),
        "vision_summary": raw_summary,
        "assistant_guidance": str(raw_analysis.get("assistant_guidance") or "").strip(),
        "key_points": normalize_analysis_list(raw_analysis.get("key_points")),
    }

    if not normalized["vision_summary"]:
        if fallback_text:
            normalized["vision_summary"] = fallback_text.strip()[:500]
        elif normalized["ocr_text"]:
            normalized["vision_summary"] = "Readable text was detected in the image and added to the context."

    has_visual_context = bool(raw_summary or normalized["key_points"] or fallback_text)
    if normalized["ocr_text"] and has_visual_context:
        normalized["assistant_guidance"] = (
            "Use the extracted OCR text as the primary image context and the visual summary for non-text cues when answering the user."
        )
    elif normalized["ocr_text"]:
        normalized["assistant_guidance"] = "Use the extracted OCR text as the primary image context when answering the user."
    elif not normalized["assistant_guidance"] and normalized["vision_summary"]:
        normalized["assistant_guidance"] = "Use the visual summary as the primary image context when answering the user."

    return normalized


def read_uploaded_image(uploaded_file):
    filename = os.path.basename((uploaded_file.filename or "").strip())
    if not filename:
        raise ValueError("Image file name is missing.")

    mime_type = (uploaded_file.mimetype or "").lower().strip()
    if mime_type not in IMAGE_ALLOWED_MIME_TYPES:
        raise ValueError("Unsupported file type. Upload PNG, JPG or WEBP.")

    image_bytes = uploaded_file.read()
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")
    if len(image_bytes) > IMAGE_MAX_BYTES:
        raise ValueError("Image is too large. Upload a maximum of 10 MB.")

    return filename, mime_type, image_bytes


def optimize_image_for_processing(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for local image optimization.") from exc

    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        longest_side = max(width, height)

        if longest_side > VISION_MAX_IMAGE_SIDE:
            scale = VISION_MAX_IMAGE_SIDE / float(longest_side)
            resized_size = (
                max(28, int(width * scale)),
                max(28, int(height * scale)),
            )
            image = image.resize(resized_size, Image.Resampling.LANCZOS)

        has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
        output = BytesIO()
        if has_alpha:
            image.save(output, format="PNG", optimize=True)
            optimized_mime_type = "image/png"
        else:
            image = image.convert("RGB")
            image.save(output, format="JPEG", quality=92, optimize=True)
            optimized_mime_type = "image/jpeg"

    optimized_bytes = output.getvalue()
    return optimized_bytes, optimized_mime_type


def run_image_vision_analysis(image_bytes: bytes, mime_type: str, user_text: str = "") -> dict:
    if not VISION_ENABLED:
        raise RuntimeError(VISION_DISABLED_FEATURE_ERROR)

    user_text = (user_text or "").strip()

    engine = get_local_vision_engine()
    torch = engine["torch"]
    model = engine["model"]
    processor = engine["processor"]
    process_vision_info = engine["process_vision_info"]
    image_bytes, mime_type = optimize_image_for_processing(image_bytes, mime_type)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_b64}"

    analysis_prompt = (
        "Analyze the image for a text-first chat assistant. Return strict JSON with exactly these keys: "
        "vision_summary, key_points, assistant_guidance. "
        "vision_summary: concise summary of important non-text visual context, written in English. "
        "key_points: array of short bullets in English with the most relevant observations, warnings, labels, UI states, numbers, or layout clues. "
        "assistant_guidance: one short sentence in English telling another LLM how to best use this image analysis when answering the user. "
        "Do not transcribe visible text from the image into the JSON because OCR is handled separately. "
        "If little visual context exists, still provide the best summary possible. "
        "Return JSON only."
    )
    if user_text:
        analysis_prompt += f" The user's current request is: {user_text}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {
                    "type": "text",
                    "text": analysis_prompt,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)

    with _vision_inference_lock:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=VISION_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_output = extract_text_from_response_content(output_text[0] if output_text else "").strip()
    parsed_output = extract_json_object(raw_output)

    del inputs
    del generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return normalize_image_analysis(parsed_output, fallback_text=raw_output)


def answer_image_question(
    image_bytes: bytes,
    mime_type: str,
    question: str,
    initial_analysis: dict | None = None,
) -> str:
    if not VISION_ENABLED:
        raise RuntimeError(VISION_DISABLED_FEATURE_ERROR)

    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ValueError("question is required.")

    engine = get_local_vision_engine()
    torch = engine["torch"]
    model = engine["model"]
    processor = engine["processor"]
    process_vision_info = engine["process_vision_info"]
    image_bytes, mime_type = optimize_image_for_processing(image_bytes, mime_type)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_b64}"

    analysis = initial_analysis if isinstance(initial_analysis, dict) else {}
    summary = str(analysis.get("vision_summary") or "").strip()
    guidance = str(analysis.get("assistant_guidance") or "").strip()
    ocr_text = str(analysis.get("ocr_text") or "").strip()

    prompt_parts = [
        "You are answering a follow-up question about a previously uploaded image.",
        "The question may come from a non-English conversation, but you must always answer in English.",
        "Use the image as the source of truth. If prior analysis exists, treat it as a hint only.",
        f"User question: {normalized_question}",
    ]
    if summary:
        prompt_parts.append(f"Initial summary hint: {summary}")
    if guidance:
        prompt_parts.append(f"Initial guidance hint: {guidance}")
    if ocr_text:
        prompt_parts.append(f"Initial OCR hint: {ocr_text[:1500]}")
    prompt_parts.append("Answer directly in English. Do not return JSON.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "\n".join(prompt_parts)},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)

    with _vision_inference_lock:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=VISION_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_output = extract_text_from_response_content(output_text[0] if output_text else "").strip()

    del inputs
    del generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not raw_output:
        raise RuntimeError("Vision model returned an empty answer.")
    return raw_output
