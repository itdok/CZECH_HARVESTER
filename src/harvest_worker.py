from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv
import edge_tts
import genanki
import requests

from src.card_generator import format_cloze_sentence, generate_cards, strip_cloze_and_html
from src.chunk_tracker import find_chunk_by_phrase, load_chunk_memory, register_harvested_phrases
from src.doc_parser import extract_harvest_phrases, generate_clear_highlight_requests

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

ANKI_CONNECT_URL = os.getenv("ANKI_CONNECT_URL", "http://localhost:8765")
DEFAULT_DECK_NAME = "Čeština::Harvester"
DEFAULT_CLOZE_MODEL = "Cloze"
EDGE_TTS_VOICE_CS = os.getenv("EDGE_VOICE_CS", "cs-CZ-VlastaNeural")

# Stable unique ID for genanki deck
GENANKI_DECK_ID = 2059394012

router = Router(name="harvest_router")


def extract_doc_id(user_input: str) -> str:
    """Extracts clean Google Docs document ID from either raw ID or URL."""
    cleaned = user_input.strip()
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", cleaned)
    if match:
        return match.group(1)
    return cleaned.split()[0] if cleaned else ""


def get_stub_document_ast(doc_id: str) -> Dict[str, Any]:
    """
    Returns a sample Google Docs AST with yellow highlights for testing/mocking
    when live Google Docs API access is not configured.
    """
    yellow_style = {
        "backgroundColor": {
            "color": {
                "rgbColor": {
                    "red": 1.0,
                    "green": 1.0,
                    "blue": 0.0,
                }
            }
        }
    }
    return {
        "documentId": doc_id,
        "title": f"Harvest Document {doc_id}",
        "body": {
            "content": [
                {
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 1,
                                "endIndex": 12,
                                "textRun": {"content": "Dnes máme "},
                            },
                            {
                                "startIndex": 12,
                                "endIndex": 23,
                                "textRun": {
                                    "content": "krásný den",
                                    "textStyle": yellow_style,
                                },
                            },
                            {
                                "startIndex": 23,
                                "endIndex": 42,
                                "textRun": {"content": " na procházku.\n"},
                            },
                        ]
                    }
                }
            ]
        },
    }


def _resolve_google_access_token(access_token: Optional[str] = None) -> Optional[str]:
    """Resolves Google OAuth access token from argument, env var, or local token.json."""
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
    if token:
        return token
    token_path = BASE_DIR / "token.json"
    if token_path.exists():
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(str(token_path))
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                token_path.write_text(creds.to_json(), encoding="utf-8")
            if creds and creds.valid and creds.token:
                return creds.token
        except Exception:
            pass
    return None


def fetch_document_ast(doc_id: str, access_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches Google Docs AST via Google Docs API v1 if token is provided,
    otherwise returns a stub AST.
    """
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
    if token:
        url = f"https://docs.googleapis.com/v1/documents/{doc_id}"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    return get_stub_document_ast(doc_id)


def clear_document_highlights(
    doc_id: str,
    clear_requests: List[Dict[str, Any]],
    access_token: Optional[str] = None,
) -> bool:
    """
    Applies UpdateTextStyleRequest batchUpdate to Google Docs API v1 to clear
    yellow highlights after phrases have been harvested.
    """
    if not clear_requests:
        return False
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN")
    if not token:
        return False
    url = f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"requests": clear_requests},
        timeout=10.0,
    )
    resp.raise_for_status()
    return True


def _ankiconnect_request(
    action: str,
    params: Optional[Dict[str, Any]] = None,
    anki_connect_url: str = ANKI_CONNECT_URL,
    timeout: float = 5.0,
) -> Any:
    """Executes an AnkiConnect v6 JSON-RPC request and validates its error field."""
    resp = requests.post(
        anki_connect_url,
        json={"action": action, "version": 6, "params": params or {}},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error") is not None:
        raise RuntimeError(f"AnkiConnect {action} error: {data.get('error')}")
    return data.get("result")


async def _synthesize_edge_tts_async(text: str, voice: str) -> bytes:
    """Streams MP3 audio bytes from edge-tts for the given Czech sentence."""
    communicate = edge_tts.Communicate(text, voice=voice)
    chunks: List[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            chunks.append(chunk["data"])
    return b"".join(chunks)


def synthesize_czech_audio(text: str, voice: str = EDGE_TTS_VOICE_CS) -> bytes:
    """Synthesizes Czech sentence MP3 audio using edge-tts, safe inside or outside an event loop."""
    if not text.strip():
        return b""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_synthesize_edge_tts_async(text, voice))
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _synthesize_edge_tts_async(text, voice)).result()


def _store_anki_media(
    clean_sentence: str,
    anki_connect_url: str = ANKI_CONNECT_URL,
) -> Optional[str]:
    """Synthesizes sentence audio via edge-tts and uploads it to Anki via storeMediaFile."""
    if not clean_sentence:
        return None
    digest = hashlib.sha256(clean_sentence.encode("utf-8")).hexdigest()[:12]
    filename = f"czech_{digest}.mp3"
    try:
        audio_bytes = synthesize_czech_audio(clean_sentence)
        if not audio_bytes:
            return None
        b64_data = base64.b64encode(audio_bytes).decode("ascii")
        _ankiconnect_request(
            "storeMediaFile",
            {"filename": filename, "data": b64_data},
            anki_connect_url=anki_connect_url,
            timeout=10.0,
        )
        return filename
    except Exception as exc:
        logger.warning("Failed to synthesize/store audio for %r: %s", clean_sentence, exc)
        return None


def _is_duplicate_in_anki(
    clean_sentence: str,
    deck_name: str,
    anki_connect_url: str = ANKI_CONNECT_URL,
) -> bool:
    """Checks Anki via findNotes using the clean sentence text without {{c1::}} syntax."""
    if not clean_sentence:
        return False
    escaped_deck = deck_name.replace('"', '\\"')
    escaped_sentence = clean_sentence.replace('"', '\\"')
    query = f'deck:"{escaped_deck}" "{escaped_sentence}"'
    result = _ankiconnect_request(
        "findNotes",
        {"query": query},
        anki_connect_url=anki_connect_url,
        timeout=3.0,
    )
    return bool(result)


def _format_back_extra(translation: str, grammar: str, sound_filename: Optional[str]) -> str:
    """Formats the Cloze Back Extra field cleanly without <br> concatenation."""
    parts: List[str] = []
    if translation:
        parts.append(f"<div>{translation}</div>")
    if grammar:
        parts.append(f"<div><i>{grammar}</i></div>")
    if sound_filename:
        parts.append(f"<div>[sound:{sound_filename}]</div>")
    return "".join(parts)


def export_via_genanki(
    cards: List[Dict[str, str]],
    deck_name: str = DEFAULT_DECK_NAME,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates an Anki package (.apkg) file using genanki's standard Cloze model."""
    target_dir = Path(output_dir) if output_dir is not None else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    deck = genanki.Deck(GENANKI_DECK_ID, deck_name)
    model = genanki.CLOZE_MODEL

    for card in cards:
        cloze_text = format_cloze_sentence(card.get("f", ""))
        back_extra = _format_back_extra(
            card.get("t", "").strip(),
            card.get("g", "").strip(),
            None,
        )
        note = genanki.Note(
            model=model,
            fields=[cloze_text, back_extra],
            tags=["czech_harvest"],
        )
        deck.add_note(note)

    date_str = datetime.date.today().isoformat()
    package_path = target_dir / f"harvest_{date_str}.apkg"

    package = genanki.Package(deck)
    package.write_to_file(str(package_path))
    return package_path


def export_to_anki(
    cards: List[Dict[str, str]],
    deck_name: str = DEFAULT_DECK_NAME,
    anki_connect_url: str = ANKI_CONNECT_URL,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Exports Cloze cards to Anki via AnkiConnect v6 (createDeck, modelFieldNames,
    findNotes, storeMediaFile, addNotes), falling back to genanki .apkg if unreachable.
    """
    if not cards:
        return {"method": "none", "count": 0, "apkg_path": None}

    # 1. Attempt AnkiConnect direct HTTP sync
    try:
        _ankiconnect_request(
            "createDeck",
            {"deck": deck_name},
            anki_connect_url=anki_connect_url,
            timeout=2.0,
        )

        field_names = _ankiconnect_request(
            "modelFieldNames",
            {"modelName": DEFAULT_CLOZE_MODEL},
            anki_connect_url=anki_connect_url,
            timeout=2.0,
        )
        if not isinstance(field_names, list) or len(field_names) < 2:
            raise RuntimeError(f"Unexpected Cloze model fields: {field_names!r}")
        first_field, second_field = field_names[0], field_names[1]

        notes_payload: List[Dict[str, Any]] = []
        skipped_duplicates = 0
        for card in cards:
            cloze_sentence = format_cloze_sentence(card.get("f", ""))
            clean_sentence = strip_cloze_and_html(cloze_sentence)

            if _is_duplicate_in_anki(clean_sentence, deck_name, anki_connect_url=anki_connect_url):
                skipped_duplicates += 1
                continue

            sound_filename = _store_anki_media(clean_sentence, anki_connect_url=anki_connect_url)
            back_extra = _format_back_extra(
                card.get("t", "").strip(),
                card.get("g", "").strip(),
                sound_filename,
            )
            notes_payload.append({
                "deckName": deck_name,
                "modelName": DEFAULT_CLOZE_MODEL,
                "fields": {
                    first_field: cloze_sentence,
                    second_field: back_extra,
                },
                "tags": ["czech_harvest"],
            })

        if not notes_payload:
            return {
                "method": "ankiconnect",
                "count": 0,
                "skipped_duplicates": skipped_duplicates,
                "apkg_path": None,
            }

        results = _ankiconnect_request(
            "addNotes",
            {"notes": notes_payload},
            anki_connect_url=anki_connect_url,
            timeout=5.0,
        )
        if isinstance(results, list):
            added_count = sum(1 for item in results if item is not None)
            if added_count > 0:
                return {
                    "method": "ankiconnect",
                    "count": added_count,
                    "skipped_duplicates": skipped_duplicates,
                    "apkg_path": None,
                }
        else:
            return {
                "method": "ankiconnect",
                "count": len(notes_payload),
                "skipped_duplicates": skipped_duplicates,
                "apkg_path": None,
            }
    except Exception:
        # AnkiConnect failed or desktop unreachable -> fallback to genanki
        pass

    # 2. Fallback to genanki .apkg export
    apkg_file = export_via_genanki(cards, deck_name=deck_name, output_dir=output_dir)
    return {
        "method": "genanki",
        "count": len(cards),
        "apkg_path": apkg_file,
    }


def run_harvest_pipeline(
    doc_id: str,
    use_local_ollama: bool = False,
    access_token: Optional[str] = None,
    output_dir: Optional[Path] = None,
    memory_path: Optional[Path] = None,
    custom_doc_ast: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates the entire harvesting workflow:
    1. Retrieve Google Docs AST.
    2. Extract highlighted Czech phrases and clear highlight requests.
    3. Generate Anki flashcards via Gemini (or local Ollama).
    4. Register harvested phrases into chunk_tracker (chunk_memory.json).
    5. Export to Anki (AnkiConnect with genanki fallback).
    """
    # 1. Fetch document AST
    doc_ast = custom_doc_ast if custom_doc_ast is not None else fetch_document_ast(doc_id, access_token=access_token)

    # 2. Extract highlighted phrases
    phrases = extract_harvest_phrases(doc_ast)
    if not phrases:
        return {
            "status": "empty",
            "doc_id": doc_id,
            "phrases_count": 0,
            "cards_count": 0,
            "export_result": {"method": "none", "count": 0, "apkg_path": None},
            "clear_requests": [],
        }

    # 3. Generate clear highlight requests
    clear_requests = generate_clear_highlight_requests(phrases)

    # 4. Filter out phrases already tracked in data/chunk_memory.json (and batch duplicates)
    memory = load_chunk_memory(memory_path)
    seen_norm: set[str] = set()
    new_phrases: List[Dict[str, Any]] = []
    for p in phrases:
        phrase_text = (p.get("phrase") or "").strip()
        if not phrase_text:
            continue
        norm = " ".join(phrase_text.lower().split())
        if norm in seen_norm or find_chunk_by_phrase(memory, phrase_text) is not None:
            continue
        seen_norm.add(norm)
        new_phrases.append(p)

    # 5. Generate cards via Gemini (or Ollama) only for non-duplicate phrases
    cards = generate_cards(new_phrases, use_local_ollama=use_local_ollama) if new_phrases else []

    # 6. Register in chunk tracker
    phrase_texts = [p["phrase"] for p in new_phrases if p.get("phrase")]
    if phrase_texts:
        register_harvested_phrases(phrase_texts, source=doc_id, memory_path=memory_path)

    # 7. Export to Anki
    export_result = export_to_anki(cards, output_dir=output_dir)

    # 7. Clear yellow highlights in Google Docs if live token is available
    highlights_cleared = False
    if custom_doc_ast is None and clear_requests:
        try:
            highlights_cleared = clear_document_highlights(
                doc_id, clear_requests, access_token=access_token
            )
        except Exception:
            highlights_cleared = False

    return {
        "status": "success",
        "doc_id": doc_id,
        "phrases_count": len(phrases),
        "cards_count": len(cards),
        "export_result": export_result,
        "clear_requests": clear_requests,
        "highlights_cleared": highlights_cleared,
        "cards": cards,
    }


@router.message(Command("harvest"))
async def handle_harvest_command(message: Message) -> None:
    """Handles the /harvest Telegram command."""
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await message.answer(
            "⚠️ Пожалуйста, укажите ID документа или ссылку:\n"
            "Пример: `/harvest 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms`",
            parse_mode="Markdown",
        )
        return

    raw_arg = parts[1].strip()
    doc_id = extract_doc_id(raw_arg)

    if not doc_id:
        await message.answer("❌ Не удалось распознать ID документа Google Docs.")
        return

    status_msg = await message.answer(f"🌾 Начинаю обработку документа `{doc_id}`...", parse_mode="Markdown")

    try:
        token = _resolve_google_access_token()
        pipeline_result = run_harvest_pipeline(doc_id, access_token=token)

        if pipeline_result["status"] == "empty":
            await status_msg.edit_text(
                f"ℹ️ В документе `{doc_id}` не найдено желтых выделений для обработки.",
                parse_mode="Markdown",
            )
            return

        export_res = pipeline_result["export_result"]
        phrases_count = pipeline_result["phrases_count"]
        cards_count = pipeline_result["cards_count"]

        if export_res["method"] == "ankiconnect":
            await status_msg.edit_text(
                f"✅ **Успешно обработано!**\n\n"
                f"• Извлечено фраз: `{phrases_count}`\n"
                f"• Создано карточек: `{cards_count}`\n"
                f"• Экспорт: отправлено напрямую в Anki через AnkiConnect.\n"
                f"• База знаний `data/chunk_memory.json` обновлена.",
                parse_mode="Markdown",
            )
        elif export_res["method"] == "genanki":
            apkg_path = export_res["apkg_path"]
            await status_msg.edit_text(
                f"⚠️ AnkiConnect недоступен. Сгенерирован файл колоды `.apkg`.\n"
                f"• Извлечено фраз: `{phrases_count}`\n"
                f"• Создано карточек: `{cards_count}`\n"
                f"• Отправляю файл...",
                parse_mode="Markdown",
            )
            input_file = FSInputFile(str(apkg_path))
            await message.answer_document(
                document=input_file,
                caption=f"📦 Колода Anki: `{apkg_path.name}` ({cards_count} карточек)",
                parse_mode="Markdown",
            )
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка при обработке документа: {e}")


def create_bot(token: Optional[str] = None) -> Bot:
    bot_token = token or os.getenv("TELEGRAM_BOT_TOKEN", "mock_token")
    return Bot(token=bot_token)


def build_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp
