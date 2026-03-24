from __future__ import annotations

import os
import threading

_embedder = None
_embedder_lock = threading.Lock()


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device() -> str:
    requested = (os.getenv("BGE_M3_DEVICE") or "").strip().lower()
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("CUDA is required for the RAG embedder, but torch could not be imported.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RAG embedder. No CUDA-capable GPU was detected.")

    if requested and requested not in {"cuda", "cuda:0"}:
        raise RuntimeError("BGE_M3_DEVICE must be set to CUDA for this application.")

    return requested or "cuda"


def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        model_name = (os.getenv("BGE_M3_MODEL_PATH") or "BAAI/bge-m3").strip()
        trust_remote_code = _parse_bool_env("BGE_M3_TRUST_REMOTE_CODE", False)
        local_files_only = _parse_bool_env("BGE_M3_LOCAL_FILES_ONLY", False) or os.path.isdir(model_name)
        device = _resolve_device()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "BGE-M3 dependencies are missing. Install sentence-transformers before using RAG."
            ) from exc

        model = SentenceTransformer(
            model_name,
            trust_remote_code=trust_remote_code,
            device=device,
            local_files_only=local_files_only,
        )
        _embedder = {
            "model": model,
            "device": device,
            "batch_size": max(1, int(os.getenv("BGE_M3_BATCH_SIZE", "32"))),
            "model_name": model_name,
            "local_files_only": local_files_only,
        }
        return _embedder


def preload_embedder() -> None:
    if not _parse_bool_env("BGE_M3_PRELOAD", True):
        return
    get_embedder()


def embed_texts(texts: list[str]) -> list[list[float]]:
    prepared = [str(text or "").strip() for text in texts if str(text or "").strip()]
    if not prepared:
        return []

    engine = get_embedder()
    vectors = engine["model"].encode(
        prepared,
        batch_size=engine["batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
