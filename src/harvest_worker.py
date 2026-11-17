from __future__ import annotations

import datetime
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv
import genanki
import requests

from src.card_generator import generate_cards
from src.chunk_tracker import register_harvested_phrases
from src.doc_parser import extract_harvest_phrases, generate_clear_highlight_requests

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

ANKI_CONNECT_URL = os.getenv("ANKI_CONNECT_URL", "http://localhost:8765")
DEFAULT_DECK_NAME = "Czech Harvesting"

# Stable unique IDs for genanki deck & model
GENANKI_MODEL_ID = 1607392319
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


def _build_genanki_model() -> genanki.Model:
    return genanki.Model(
        GENANKI_MODEL_ID,
        "Czech Harvesting Mining Model",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
            {"name": "Grammar"},
        ],
        templates=[
            {
                "name": "Czech Sentence Mining Card",
                "qfmt": "{{Front}}",
                "afmt": (
                    "{{FrontSide}}<hr id=\"answer\">"
                    "<div style=\"font-size: 1.2em; font-weight: bold;\">{{Back}}</div><br>"
                    "<div style=\"color: #666; font-size: 0.9em; font-style: italic;\">{{Grammar}}</div>"
                ),
            }
        ],
        css=(
            ".card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
            "font-size: 20px; text-align: center; color: #222; background-color: #fff; padding: 20px; }\n"
            "b { color: #d9534f; font-weight: bold; }\n"
        ),
    )


def export_via_genanki(
    cards: List[Dict[str, str]],
    deck_name: str = DEFAULT_DECK_NAME,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generates an Anki package (.apkg) file using genanki."""
    target_dir = Path(output_dir) if output_dir is not None else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    deck = genanki.Deck(GENANKI_DECK_ID, deck_name)
    model = _build_genanki_model()

    for card in cards:
        note = genanki.Note(
            model=model,
            fields=[
                card.get("f", ""),
                card.get("t", ""),
                card.get("g", ""),
            ],
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
    Exports cards to Anki.
    First attempts HTTP POST to local AnkiConnect (http://localhost:8765).
    If AnkiConnect is unreachable or returns an error, falls back to genanki .apkg generator.
    """
    if not cards:
        return {"method": "none", "count": 0, "apkg_path": None}

    # 1. Attempt AnkiConnect direct HTTP sync
    try:
        # Create deck if not exists
        deck_resp = requests.post(
            anki_connect_url,
            json={"action": "createDeck", "version": 6, "params": {"deck": deck_name}},
            timeout=2.0,
        )
        deck_resp.raise_for_status()
        deck_data = deck_resp.json()
        if deck_data.get("error") is not None:
            raise RuntimeError(f"AnkiConnect createDeck error: {deck_data.get('error')}")

        notes_payload = []
        for card in cards:
            notes_payload.append({
                "deckName": deck_name,
                "modelName": "Basic",
                "fields": {
                    "Front": card.get("f", ""),
                    "Back": f"{card.get('t', '')}<br><br><small><i>{card.get('g', '')}</i></small>",
                },
                "tags": ["czech_harvest"],
            })

        resp = requests.post(
            anki_connect_url,
            json={"action": "addNotes", "version": 6, "params": {"notes": notes_payload}},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("error") is None:
            results = data.get("result")
            if isinstance(results, list):
                added_count = sum(1 for item in results if item is not None)
                if added_count > 0:
                    return {
                        "method": "ankiconnect",
                        "count": added_count,
                        "apkg_path": None,
                    }
            else:
                return {
                    "method": "ankiconnect",
                    "count": len(cards),
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

    # 4. Generate cards via Gemini (or Ollama)
    cards = generate_cards(phrases, use_local_ollama=use_local_ollama)

    # 5. Register in chunk tracker
    phrase_texts = [p["phrase"] for p in phrases if p.get("phrase")]
    register_harvested_phrases(phrase_texts, source=doc_id, memory_path=memory_path)

    # 6. Export to Anki
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
