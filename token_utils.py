from __future__ import annotations

import re
from functools import lru_cache

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency fallback
    tiktoken = None


@lru_cache(maxsize=1)
def get_token_encoder():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_text_tokens(text: str) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 0

    encoder = get_token_encoder()
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(normalized, disallowed_special=())))
        except Exception:
            pass

    byte_estimate = (len(normalized.encode("utf-8")) + 3) // 4
    piece_estimate = len(re.findall(r"\w+|[^\w\s]", normalized, re.UNICODE))
    return max(1, byte_estimate, piece_estimate)