from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_MEMORY_PATH = BASE_DIR / "data" / "chunk_memory.json"

DEFAULT_CHUNK_SCHEMA_VERSION = "1.0"


class ChunkMemoryValidationError(Exception):
    pass


def _today() -> str:
    return datetime.date.today().isoformat()


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _clamp_float(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return round(max(min_value, min(max_value, value)), 2)


def _normalize_string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return str(value)


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    normalized: List[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)

    return normalized


def _normalize_phrase(phrase: str) -> str:
    """Normalizes whitespace and casing for canonical ID computation and lookup."""
    if not isinstance(phrase, str):
        return ""
    return " ".join(phrase.strip().lower().split())


def generate_chunk_id(phrase: str) -> str:
    """
    Generates a deterministic unique chunk ID based on the normalized Czech phrase.
    Format: chunk_<sha256_hash_12_chars>
    """
    norm = _normalize_phrase(phrase)
    if not norm:
        raise ChunkMemoryValidationError("Cannot generate chunk_id for empty phrase")
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"


# Alias for backward/forward compatibility
compute_chunk_id = generate_chunk_id


def _calculate_interval(weight: float, usage_count: int) -> int:
    """Calculates spaced repetition review interval in days based on weight and usage."""
    if usage_count <= 0:
        return 1
    if weight <= 0.2:
        return 1
    elif weight <= 0.4:
        return 3
    elif weight <= 0.6:
        return 7
    elif weight <= 0.8:
        return 14
    else:
        return 30


def _default_chunk_record(phrase: str = "", initial_weight: float = 0.0) -> Dict[str, Any]:
    today = _today()
    w = _clamp_float(initial_weight, 0.0, 1.0)
    clean_phrase = phrase.strip() if isinstance(phrase, str) else ""
    return {
        "schema_version": DEFAULT_CHUNK_SCHEMA_VERSION,
        "phrase": clean_phrase,
        "level": w,
        "weight": w,
        "success_rate": 0.0,
        "first_seen": today,
        "last_seen": today,
        "usage_count": 0,
        "interval_days": 1,
        "source_lesson": None,
        "variation_forms": [clean_phrase] if clean_phrase else [],
    }


def _sanitize_chunk_record(raw_record: Any, default_phrase: str = "") -> Dict[str, Any]:
    today = _today()

    if not isinstance(raw_record, dict):
        raw_record = {}

    record = _default_chunk_record()

    schema_version = raw_record.get("schema_version", DEFAULT_CHUNK_SCHEMA_VERSION)
    record["schema_version"] = str(schema_version).strip() or DEFAULT_CHUNK_SCHEMA_VERSION

    # Phrase resolution: prefer explicit phrase, then default_phrase, then first variation_form
    phrase = raw_record.get("phrase")
    if not phrase and default_phrase:
        phrase = default_phrase
    if not phrase and isinstance(raw_record.get("variation_forms"), list) and raw_record["variation_forms"]:
        phrase = raw_record["variation_forms"][0]
    record["phrase"] = str(phrase).strip() if phrase else ""

    record["usage_count"] = max(0, _safe_int(raw_record.get("usage_count", 0), 0))
    record["weight"] = _clamp_float(_safe_float(raw_record.get("weight", 0.0), 0.0))
    record["success_rate"] = _clamp_float(_safe_float(raw_record.get("success_rate", 0.0), 0.0))
    record["level"] = _clamp_float(_safe_float(raw_record.get("level", record["weight"]), record["weight"]))

    default_interval = _calculate_interval(record["weight"], record["usage_count"])
    record["interval_days"] = max(1, _safe_int(raw_record.get("interval_days", default_interval), default_interval))

    first_seen = raw_record.get("first_seen", today)
    last_seen = raw_record.get("last_seen", today)

    record["first_seen"] = first_seen if isinstance(first_seen, str) and first_seen.strip() else today
    record["last_seen"] = last_seen if isinstance(last_seen, str) and last_seen.strip() else today

    record["source_lesson"] = _normalize_string_or_none(raw_record.get("source_lesson"))
    record["variation_forms"] = _normalize_string_list(raw_record.get("variation_forms", []))
    if record["phrase"] and record["phrase"] not in record["variation_forms"]:
        record["variation_forms"].insert(0, record["phrase"])

    return record


def load_chunk_memory(path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Loads and sanitizes chunk memory from JSON file.
    Does not restrict to any static registry; accepts all valid records.
    """
    file_path = Path(path) if path is not None else CHUNK_MEMORY_PATH
    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ChunkMemoryValidationError(f"Failed to read chunk_memory.json at {file_path}: {e}") from e

    if not isinstance(data, dict):
        return {}

    sanitized_memory: Dict[str, Any] = {}
    for chunk_id, raw_record in data.items():
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        sanitized_memory[chunk_id.strip()] = _sanitize_chunk_record(raw_record)

    return sanitized_memory


def save_chunk_memory(memory: Dict[str, Any], path: Optional[Union[Path, str]] = None) -> None:
    """
    Persists chunk memory dictionary to disk as formatted JSON.
    """
    file_path = Path(path) if path is not None else CHUNK_MEMORY_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)

    sanitized_to_save: Dict[str, Any] = {}
    for chunk_id, raw_record in memory.items():
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        sanitized_to_save[chunk_id.strip()] = _sanitize_chunk_record(raw_record)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_to_save, f, ensure_ascii=False, indent=2)


def find_chunk_by_phrase(memory: Dict[str, Any], phrase: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Finds a chunk in memory by matching phrase text (case-insensitive and normalized).
    """
    norm = _normalize_phrase(phrase)
    if not norm:
        return None

    # Check direct hash ID match first
    candidate_id = generate_chunk_id(phrase)
    if candidate_id in memory:
        return candidate_id, memory[candidate_id]

    # Search existing records for phrase or variations
    for chunk_id, record in memory.items():
        if _normalize_phrase(record.get("phrase", "")) == norm:
            return chunk_id, record
        for var in record.get("variation_forms", []):
            if _normalize_phrase(var) == norm:
                return chunk_id, record

    return None


def resolve_chunk_id(memory: Dict[str, Any], chunk_or_phrase: str) -> str:
    """
    Resolves chunk_id from an input string:
    - If it's already an existing chunk_id in memory, returns it.
    - If it matches an existing phrase in memory, returns its chunk_id.
    - If it has chunk ID prefix format (e.g. 'chunk_' or 'chk_'), treats it as chunk_id.
    - Otherwise generates a hash-based chunk_id for the dynamic phrase.
    """
    cleaned = chunk_or_phrase.strip()
    if not cleaned:
        raise ChunkMemoryValidationError("chunk_or_phrase cannot be empty")

    if cleaned in memory:
        return cleaned

    found = find_chunk_by_phrase(memory, cleaned)
    if found:
        return found[0]

    if cleaned.startswith("chunk_") or cleaned.startswith("chk_"):
        return cleaned

    return generate_chunk_id(cleaned)


def ensure_chunk_record(
    memory: Dict[str, Any],
    chunk_id_or_phrase: str,
    phrase: Optional[str] = None,
    initial_weight: float = 0.0,
) -> str:
    """
    Ensures that a chunk record exists in the memory dictionary.
    Accepts arbitrary Czech phrases or explicit chunk IDs.
    Returns the resolved chunk_id.
    """
    if not isinstance(chunk_id_or_phrase, str) or not chunk_id_or_phrase.strip():
        raise ChunkMemoryValidationError("chunk_id_or_phrase must be a non-empty string")

    chunk_id = resolve_chunk_id(memory, chunk_id_or_phrase)

    # Determine phrase string
    record_phrase = phrase
    if not record_phrase:
        if not (chunk_id_or_phrase.startswith("chunk_") or chunk_id_or_phrase.startswith("chk_")):
            record_phrase = chunk_id_or_phrase.strip()
        else:
            record_phrase = ""

    if chunk_id not in memory:
        memory[chunk_id] = _default_chunk_record(phrase=record_phrase, initial_weight=initial_weight)
    else:
        memory[chunk_id] = _sanitize_chunk_record(memory[chunk_id], default_phrase=record_phrase)

    return chunk_id


def update_chunk_on_review(memory: Dict[str, Any], chunk_id_or_phrase: str, success: bool = True) -> str:
    """
    Updates chunk statistics based on review outcome.
    - success=True increases weight (+0.1) and contributes 1.0 to success_rate.
    - success=False decreases weight (-0.1) and contributes 0.0 to success_rate.
    """
    chunk_id = ensure_chunk_record(memory, chunk_id_or_phrase)
    today = _today()
    record = memory[chunk_id]

    usage_count = max(0, _safe_int(record.get("usage_count", 0), 0)) + 1
    old_weight = _clamp_float(_safe_float(record.get("weight", 0.0), 0.0))
    old_success_rate = _clamp_float(_safe_float(record.get("success_rate", 0.0), 0.0))

    if success:
        new_weight = min(1.0, old_weight + 0.1)
        outcome_score = 1.0
    else:
        new_weight = max(0.0, old_weight - 0.1)
        outcome_score = 0.0

    if usage_count <= 1:
        new_success_rate = outcome_score
    else:
        new_success_rate = ((old_success_rate * (usage_count - 1)) + outcome_score) / usage_count

    record["schema_version"] = DEFAULT_CHUNK_SCHEMA_VERSION
    record["usage_count"] = usage_count
    record["last_seen"] = today
    record["weight"] = round(new_weight, 2)
    record["success_rate"] = round(new_success_rate, 2)
    record["level"] = round(new_weight, 2)
    record["interval_days"] = _calculate_interval(record["weight"], usage_count)

    if not record.get("first_seen"):
        record["first_seen"] = today

    return chunk_id


def update_chunk_on_success(memory: Dict[str, Any], chunk_id_or_phrase: str) -> str:
    """
    Updates chunk statistics on successful practice or harvest occurrence.
    """
    return update_chunk_on_review(memory, chunk_id_or_phrase, success=True)


def register_phrase(
    phrase: str,
    source_lesson: Optional[str] = None,
    initial_weight: float = 0.0,
    memory_path: Optional[Union[Path, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Registers a dynamic Czech phrase into chunk memory.
    Persists immediately to data/chunk_memory.json (or specified memory_path).
    Returns (chunk_id, record).
    """
    memory = load_chunk_memory(path=memory_path)
    chunk_id = ensure_chunk_record(memory, phrase, initial_weight=initial_weight)
    if source_lesson:
        memory[chunk_id]["source_lesson"] = source_lesson
    save_chunk_memory(memory, path=memory_path)
    return chunk_id, memory[chunk_id]


def register_used_chunks(
    chunk_ids_or_phrases: List[str],
    memory_path: Optional[Union[Path, str]] = None,
    source_lesson: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Accepts arbitrary Czech phrases or chunk IDs, increments practice count,
    updates weights/intervals, and persists to data/chunk_memory.json.
    """
    memory = load_chunk_memory(path=memory_path)

    for item in chunk_ids_or_phrases:
        if not isinstance(item, str):
            continue

        cleaned = item.strip()
        if not cleaned:
            continue

        chunk_id = update_chunk_on_success(memory, cleaned)
        if source_lesson and memory[chunk_id].get("source_lesson") is None:
            memory[chunk_id]["source_lesson"] = source_lesson

    save_chunk_memory(memory, path=memory_path)
    return memory


def register_harvested_phrases(
    phrases: List[str],
    source: Optional[str] = None,
    memory_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for harvested phrases from Google Docs or audiobooks.
    """
    return register_used_chunks(phrases, memory_path=memory_path, source_lesson=source)


def get_chunk(memory: Dict[str, Any], chunk_id_or_phrase: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves chunk record by chunk_id or by Czech phrase text.
    """
    cleaned = chunk_or_phrase = chunk_id_or_phrase.strip()
    if cleaned in memory:
        return memory[cleaned]

    found = find_chunk_by_phrase(memory, cleaned)
    if found:
        return found[1]

    return None


if __name__ == "__main__":
    test_chunks = ["chunk_004", "mít se dobře", "drahá kniha"]
    updated_memory = register_used_chunks(test_chunks)

    print("=== UPDATED CHUNK MEMORY ===")
    print(json.dumps(updated_memory, ensure_ascii=False, indent=2))
