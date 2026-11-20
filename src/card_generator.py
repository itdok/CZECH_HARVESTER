from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import requests

load_dotenv()


class CardItem(BaseModel):
    f: str = Field(description="Czech sentence/context with target phrase bolded in <b>...</b> tags")
    t: str = Field(description="Russian translation of the sentence and target phrase")
    g: str = Field(description="Brief grammar/morphology notes in Russian (case, aspect, form, etc.)")
    h: str = Field(
        default="",
        description="Short Russian translation of ONLY the hidden Czech target phrase (2-5 words, no full sentence, no colons)",
    )


class CardBatch(BaseModel):
    cards: List[CardItem] = Field(description="List of generated flashcard items")


class CardGenerationError(Exception):
    """Exception raised when flashcard generation fails."""
    pass


def strip_cloze_and_html(text: str) -> str:
    """Removes Anki Cloze syntax ({{c1::chunk}} or {{c1::chunk::hint}}) and basic HTML bold tags to get clean text."""
    if not text:
        return ""
    cleaned = re.sub(r"\{\{c\d+::((?:(?!::|\}\}).)*)(?:::[^}]*)?\}\}", r"\1", text)
    cleaned = re.sub(r"</?b>", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def _sanitize_cloze_hint(hint: str) -> str:
    """Normalizes a Cloze hint and strips colons so Anki Cloze syntax is never broken."""
    if not hint:
        return ""
    return " ".join(hint.replace(":", "").split()).strip()


def format_cloze_sentence(f_text: str, target_phrase: str = "", hint: str = "") -> str:
    """
    Formats a Czech sentence for Anki Cloze:
    - Converts <b>...</b> tags to {{c1::...}} (or {{c1::...::hint}} when hint is provided).
    - Adds ::hint to existing {{c1::...}} without a hint if hint is provided.
    - Otherwise wraps target_phrase in {{c1::...}} (case-insensitive).
    """
    clean_hint = _sanitize_cloze_hint(hint)
    hint_suffix = f"::{clean_hint}" if clean_hint else ""

    if not f_text:
        return f"{{{{c1::{target_phrase}{hint_suffix}}}}}" if target_phrase else ""
    if re.search(r"\{\{c\d+::", f_text):
        if clean_hint:
            # Add hint only to clozes that do not already have a ::hint
            return re.sub(
                r"\{\{(c\d+)::((?:(?!::|\}\}).)*)\}\}",
                lambda m: f"{{{{{m.group(1)}::{m.group(2)}{hint_suffix}}}}}",
                f_text,
            )
        return f_text
    if re.search(r"<b>.*?</b>", f_text, flags=re.IGNORECASE):
        return re.sub(
            r"<b>(.*?)</b>",
            lambda m: f"{{{{c1::{m.group(1)}{hint_suffix}}}}}",
            f_text,
            flags=re.IGNORECASE,
        )

    clean_phrase = target_phrase.strip()
    if clean_phrase:
        match = re.search(re.escape(clean_phrase), f_text, flags=re.IGNORECASE)
        if match:
            start, end = match.span()
            return f"{f_text[:start]}{{{{c1::{f_text[start:end]}{hint_suffix}}}}}{f_text[end:]}"
        return f"{f_text} ({{{{c1::{clean_phrase}{hint_suffix}}}}})"

    return f_text


def _ensure_bold_phrase(f_text: str, target_phrase: str) -> str:
    """
    Ensures that target_phrase is highlighted with <b>...</b> tags inside f_text.
    If <b> tags are already present, leaves f_text unchanged.
    """
    if not f_text:
        return f"<b>{target_phrase}</b>" if target_phrase else ""
    if "<b>" in f_text and "</b>" in f_text:
        return f_text

    clean_phrase = target_phrase.strip()
    if clean_phrase:
        pattern = re.compile(re.escape(clean_phrase), re.IGNORECASE)
        match = pattern.search(f_text)
        if match:
            start, end = match.span()
            return f"{f_text[:start]}<b>{f_text[start:end]}</b>{f_text[end:]}"
        return f"{f_text} (<b>{clean_phrase}</b>)"

    return f_text


def _build_prompt(phrases: List[Dict[str, Any]]) -> str:
    """Builds instructions and batch input for Gemini/Ollama prompt."""
    formatted_items: List[str] = []
    for idx, p in enumerate(phrases, 1):
        phrase_text = p.get("phrase", "").strip()
        context_text = p.get("context", "").strip()
        if not context_text:
            context_text = phrase_text
        formatted_items.append(f"{idx}. Target: '{phrase_text}' | Full context: '{context_text}'")

    input_block = "\n".join(formatted_items)
    return (
        "You are an expert Czech language tutor for Russian speakers.\n"
        "Generate spaced repetition flashcards for each of the following harvested Czech phrases.\n\n"
        "Requirements for each card:\n"
        "- `f`: The complete Czech sentence from context with the target phrase wrapped in <b>...</b> tags "
        "(e.g., 'To je <b>drahá kniha</b>.').\n"
        "- `t`: Natural Russian translation of the sentence and target phrase in this specific context.\n"
        "- `g`: Brief, precise grammatical note in Russian (part of speech, case/pád, aspect/vid dokon./nedokon., "
        "preposition government, or key usage pattern).\n"
        "- `h`: Short Russian translation of ONLY the hidden Czech target phrase (2–5 words, no full sentence "
        "translation, no colons) to be used as an Anki Cloze hint.\n\n"
        f"Phrases:\n{input_block}"
    )


def _finalize_cards(card_items: List[CardItem], original_phrases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Finalizes card list ensuring bold tags on front text."""
    result: List[Dict[str, str]] = []
    for idx, card in enumerate(card_items):
        target_phrase = original_phrases[idx].get("phrase", "") if idx < len(original_phrases) else ""
        f_text = _ensure_bold_phrase(card.f, target_phrase)
        result.append({
            "f": f_text,
            "t": card.t.strip(),
            "g": card.g.strip(),
            "h": _sanitize_cloze_hint(card.h),
        })
    return result


def _parse_card_batch_response(response: Any) -> CardBatch:
    """Extracts and validates CardBatch from google-genai response object."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, CardBatch):
            return parsed
        if isinstance(parsed, dict):
            return CardBatch.model_validate(parsed)

    text = getattr(response, "text", None)
    if text:
        return CardBatch.model_validate_json(text)

    raise CardGenerationError("Empty or unparsable response from Gemini API")


def _generate_cards_gemini(
    phrases: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """Calls Google Gemini API using structured outputs."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key and client is None:
        raise CardGenerationError("GEMINI_API_KEY is not set in environment or arguments.")

    model = model_name or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    ai_client = (
        client
        if client is not None
        else genai.Client(api_key=key, http_options={"api_version": "v1beta"})
    )

    prompt = _build_prompt(phrases)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CardBatch,
    )

    try:
        response = ai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        raise CardGenerationError(f"Gemini API generation failed: {e}") from e

    card_batch = _parse_card_batch_response(response)
    return _finalize_cards(card_batch.cards, phrases)


def _generate_cards_ollama(
    phrases: List[Dict[str, Any]],
    endpoint: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: float = 30.0,
) -> List[Dict[str, str]]:
    """Calls local Ollama router endpoint using structured JSON schema."""
    url = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    prompt = _build_prompt(phrases)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": CardBatch.model_json_schema(),
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise CardGenerationError(f"Ollama local router request failed: {e}") from e

    raw_content = data.get("response")
    if raw_content is None and "message" in data:
        raw_content = data["message"].get("content")

    if not raw_content:
        raise CardGenerationError(f"Empty response from Ollama endpoint: {data}")

    try:
        if isinstance(raw_content, dict):
            batch = CardBatch.model_validate(raw_content)
        else:
            batch = CardBatch.model_validate_json(raw_content)
    except Exception as e:
        raise CardGenerationError(f"Failed to validate Ollama response into CardBatch: {e}") from e

    return _finalize_cards(batch.cards, phrases)


def generate_cards(
    phrases: List[Dict[str, Any]],
    use_local_ollama: bool = False,
    api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    ollama_endpoint: Optional[str] = None,
    ollama_model: Optional[str] = None,
    client: Optional[Any] = None,
    fallback_to_ollama: bool = False,
) -> List[Dict[str, str]]:
    """
    Generates flashcards for Czech phrases using either Google Gemini API or local Ollama endpoint.
    Output structure for each card:
    - 'f': Front Czech sentence with target phrase bolded via <b>...</b>
    - 't': Russian translation
    - 'g': Brief grammatical note in Russian
    """
    valid_phrases = [p for p in phrases if isinstance(p, dict) and p.get("phrase", "").strip()]
    if not valid_phrases:
        return []

    if use_local_ollama:
        return _generate_cards_ollama(
            valid_phrases,
            endpoint=ollama_endpoint,
            model_name=ollama_model,
        )

    try:
        return _generate_cards_gemini(
            valid_phrases,
            api_key=api_key,
            model_name=gemini_model,
            client=client,
        )
    except Exception as e:
        if fallback_to_ollama:
            return _generate_cards_ollama(
                valid_phrases,
                endpoint=ollama_endpoint,
                model_name=ollama_model,
            )
        raise
