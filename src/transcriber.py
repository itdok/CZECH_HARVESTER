from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from faster_whisper import WhisperModel


CZECH_LONG_VOWELS: Set[str] = {
    "á", "é", "í", "ó", "ú", "ů", "ý",
    "Á", "É", "Í", "Ó", "Ú", "Ů", "Ý",
}

HIGHLIGHT_COLOR_RGB = {
    "red": 0.85,
    "green": 0.32,
    "blue": 0.31,
}


def transcribe_audio(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    model: Optional[WhisperModel] = None,
    **kwargs: Any,
) -> str:
    """
    Transcribes audio file to clean Czech text using faster-whisper.
    Defaults to large-v3 model on CUDA with float16 precision and language='cs'.
    """
    if not audio_path or not str(audio_path).strip():
        raise ValueError("audio_path must be a non-empty string")

    if model is None:
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            # Fallback to CPU if CUDA is requested but unavailable on host
            if device == "cuda":
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
            else:
                raise

    segments, _ = model.transcribe(
        audio_path,
        language="cs",
        **kwargs,
    )

    text_parts: List[str] = []
    for segment in segments:
        text = getattr(segment, "text", "")
        if text and text.strip():
            text_parts.append(text.strip())

    return " ".join(text_parts).strip()


def generate_vowel_highlight_requests(
    text: str,
    document_start_index: int = 1,
) -> List[Dict[str, Any]]:
    """
    Finds all Czech long vowels (á, é, í, ó, ú, ů, ý and uppercase counterparts)
    and generates UpdateTextStyleRequest dicts for Google Docs API batchUpdate.
    
    Applies bold: True and red text color (RGB: 0.85, 0.32, 0.31).
    Accurately computes startIndex and endIndex based on UTF-16 code units.
    """
    if not text:
        return []

    requests: List[Dict[str, Any]] = []
    current_utf16_offset = 0

    for char in text:
        # UTF-16 code units: characters in BMP (ord <= 0xFFFF) take 1 unit,
        # astral plane characters (surrogate pairs, e.g. emojis) take 2 units.
        char_utf16_len = 2 if ord(char) > 0xFFFF else 1

        if char in CZECH_LONG_VOWELS:
            start_idx = document_start_index + current_utf16_offset
            end_idx = start_idx + char_utf16_len

            request = {
                "updateTextStyle": {
                    "range": {
                        "startIndex": start_idx,
                        "endIndex": end_idx,
                    },
                    "textStyle": {
                        "bold": True,
                        "foregroundColor": {
                            "color": {
                                "rgbColor": {
                                    "red": HIGHLIGHT_COLOR_RGB["red"],
                                    "green": HIGHLIGHT_COLOR_RGB["green"],
                                    "blue": HIGHLIGHT_COLOR_RGB["blue"],
                                }
                            }
                        },
                    },
                    "fields": "bold,foregroundColor",
                }
            }
            requests.append(request)

        current_utf16_offset += char_utf16_len

    return requests
