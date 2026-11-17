from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, List, Optional, Set, Tuple

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv
import edge_tts
from faster_whisper import WhisperModel

from src.session_manager import SessionManager

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = BASE_DIR / "data" / "tmp"

WHISPER_AGENT_MODEL = os.getenv("WHISPER_AGENT_MODEL", "small")
EDGE_VOICE_RU = os.getenv("EDGE_VOICE_RU", "ru-RU-SvetlanaNeural")
EDGE_VOICE_CS = os.getenv("EDGE_VOICE_CS", "cs-CZ-VlastaNeural")
EDGE_VOICE_EN = os.getenv("EDGE_VOICE_EN", "en-US-AriaNeural")

CZECH_DIACRITICS = set("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")

_whisper_model: Optional[WhisperModel] = None
_voice_replies_enabled: bool = True

agent_router = Router(name="voice_agent_router")
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_allowed_user_ids() -> Set[int]:
    raw = os.getenv("ALLOWED_TELEGRAM_USER_ID", "").strip()
    if not raw:
        return set()
    ids: Set[int] = set()
    for part in raw.split(","):
        cleaned = part.strip()
        if cleaned.isdigit():
            ids.add(int(cleaned))
    return ids


def is_user_allowed(user_id: Optional[int]) -> bool:
    allowed = get_allowed_user_ids()
    if not allowed:
        return True
    return user_id in allowed


def _get_whisper_model(force_cpu: bool = False) -> WhisperModel:
    global _whisper_model
    if force_cpu or _whisper_model is None:
        device = "cpu" if force_cpu else os.getenv("WHISPER_DEVICE", "cpu")
        if device == "cuda":
            try:
                _whisper_model = WhisperModel(
                    WHISPER_AGENT_MODEL, device="cuda", compute_type="float16"
                )
            except Exception:
                _whisper_model = WhisperModel(
                    WHISPER_AGENT_MODEL, device="cpu", compute_type="int8"
                )
        else:
            _whisper_model = WhisperModel(
                WHISPER_AGENT_MODEL, device="cpu", compute_type="int8"
            )
    return _whisper_model


def _decode_audio_to_16k_mono(audio_path: str):
    """Decodes any audio file (ogg/mp3/wav) to 16kHz mono float32 numpy array via PyAV."""
    import av
    import numpy as np

    chunks = []
    with av.open(audio_path, mode="r") as container:
        in_stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in container.decode(in_stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                chunks.append(arr.reshape(-1))
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            chunks.append(arr.reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def transcribe_voice_file(audio_path: str, model: Optional[WhisperModel] = None) -> str:
    """
    Transcribes a local voice file using faster-whisper with automatic language detection
    so Russian, Czech, and English commands all work seamlessly.
    """
    whisper = model or _get_whisper_model()
    audio_input: Any = audio_path
    if os.path.exists(audio_path):
        audio_input = _decode_audio_to_16k_mono(audio_path)

    try:
        segments, _ = whisper.transcribe(
            audio_input, language=None, vad_filter=True, beam_size=1
        )
    except RuntimeError as exc:
        if model is None and ("cublas" in str(exc).lower() or "cuda" in str(exc).lower()):
            logger.warning("CUDA runtime unavailable (%s), switching Whisper to CPU int8", exc)
            whisper = _get_whisper_model(force_cpu=True)
            segments, _ = whisper.transcribe(
                audio_input, language=None, vad_filter=True, beam_size=1
            )
        else:
            raise

    parts: List[str] = []
    for seg in segments:
        text = getattr(seg, "text", "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def detect_tts_voice(text: str) -> str:
    """Selects Russian (default, handles English tech terms too) or Czech Edge TTS voice."""
    cyrillic_count = len(CYRILLIC_RE.findall(text))
    czech_count = sum(1 for ch in text if ch in CZECH_DIACRITICS)
    if czech_count >= 1 and cyrillic_count < 3:
        return EDGE_VOICE_CS
    return EDGE_VOICE_RU


def split_bilingual_segments(text: str) -> List[Tuple[str, str]]:
    """
    Splits mixed Russian/Czech text into contiguous segments paired with
    their native Edge TTS voice (ru-RU-SvetlanaNeural vs cs-CZ-VlastaNeural).
    English/ASCII tech words (Gemini, Telegram, test.txt) stay in the Russian
    stream so we avoid extra TTS calls and choppy voice switches.
    """
    if not text.strip():
        return []

    # Group contiguous Latin+punctuation runs vs Cyrillic+punctuation runs
    latin_run_re = re.compile(
        r"([a-zA-ZáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-zA-ZáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ0-9\s.,!?;:\-\"'«»()]*[a-zA-ZáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ.,!?]?)"
    )

    segments: List[Tuple[str, str]] = []
    pos = 0
    for m in latin_run_re.finditer(text):
        start, end = m.span()
        latin_chunk = m.group(0)
        # Only switch to Czech voice if the Latin phrase actually has Czech diacritics
        if any(ch in CZECH_DIACRITICS for ch in latin_chunk):
            before = text[pos:start].strip()
            if before:
                segments.append((before, EDGE_VOICE_RU))
            cz_clean = latin_chunk.strip()
            if cz_clean:
                segments.append((cz_clean, EDGE_VOICE_CS))
            pos = end

    tail = text[pos:].strip()
    if tail:
        segments.append((tail, EDGE_VOICE_RU))

    if not segments:
        return [(text.strip(), detect_tts_voice(text))]

    # Merge adjacent segments that share the same voice and drop punctuation-only segments
    merged: List[Tuple[str, str]] = []
    has_alnum = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")
    for seg_text, seg_voice in segments:
        if not has_alnum.search(seg_text):
            continue
        if merged and merged[-1][1] == seg_voice:
            merged[-1] = (f"{merged[-1][0]} {seg_text}", seg_voice)
        else:
            merged.append((seg_text, seg_voice))
    return merged


def clean_text_for_tts(text: str) -> str:
    """Strips markdown and emoji/non-speech symbols so TTS reads responses naturally."""
    cleaned = re.sub(r"```[\s\S]*?```", " код ", text)
    cleaned = re.sub(r"[`*_#>]", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Keep letters, digits, whitespace, and standard punctuation
    cleaned = re.sub(r"[^\w\s.,!?;:\-—()\"'«»áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽа-яА-ЯёЁ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:2500]


def split_voice_chunks(text: str, first_chunk_max: int = 320) -> List[str]:
    """
    Splits a long reply into an immediate first chunk (at a sentence boundary)
    and the remaining text so the user hears the start of the answer right away.
    """
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return []
    if len(cleaned) <= first_chunk_max + 80:
        return [cleaned]

    # Find a sentence boundary near first_chunk_max
    split_idx = -1
    for m in re.finditer(r"[.!?]\s+", cleaned):
        end_pos = m.end()
        if end_pos >= 80:
            split_idx = end_pos
            if end_pos >= first_chunk_max:
                break

    if split_idx <= 0 or split_idx >= len(cleaned) - 20:
        return [cleaned]

    first = cleaned[:split_idx].strip()
    rest = cleaned[split_idx:].strip()
    return [c for c in (first, rest) if c]


def _concat_audio_to_ogg_opus(src_paths: List[Path], ogg_path: Path) -> bool:
    """Concatenates one or more MP3/WAV files into a single OGG Opus file using PyAV."""
    try:
        import av

        with av.open(str(ogg_path), mode="w", format="ogg") as out_container:
            out_stream = out_container.add_stream("libopus", rate=48000)
            out_stream.bit_rate = 64000
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=48000,
            )
            for src_path in src_paths:
                if not src_path.exists() or src_path.stat().st_size == 0:
                    continue
                with av.open(str(src_path)) as in_container:
                    in_stream = in_container.streams.audio[0]
                    for frame in in_container.decode(in_stream):
                        for resampled in resampler.resample(frame):
                            resampled.pts = None
                            for packet in out_stream.encode(resampled):
                                out_container.mux(packet)
            for resampled in resampler.resample(None):
                resampled.pts = None
                for packet in out_stream.encode(resampled):
                    out_container.mux(packet)
            for packet in out_stream.encode(None):
                out_container.mux(packet)
        return ogg_path.exists() and ogg_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("PyAV audio->OGG conversion failed: %s", exc)
        return False


def _synthesize_windows_sapi5(text: str, wav_path: Path) -> bool:
    """Offline Windows SAPI5 fallback TTS via PowerShell System.Speech."""
    import subprocess

    safe_text = text.replace("'", "''")
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{wav_path}'); "
        f"$s.Speak('{safe_text}'); "
        "$s.Dispose()"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("Windows SAPI5 fallback failed: %s", exc)
        return False


async def synthesize_voice_file(text: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Synthesizes speech using edge-tts per language segment (Russian vs Czech)
    so Czech phrases inside Russian answers are pronounced natively,
    then concatenates into a single OGG Opus voice message.
    """
    spoken_text = clean_text_for_tts(text)
    if not spoken_text:
        return None

    target_dir = output_dir or TMP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    segments = split_bilingual_segments(spoken_text)
    if not segments:
        segments = [(spoken_text, detect_tts_voice(spoken_text))]

    part_paths: List[Path] = []
    ogg_fd, ogg_name = tempfile.mkstemp(suffix=".ogg", dir=target_dir)
    os.close(ogg_fd)
    ogg_path = Path(ogg_name)
    wav_path = ogg_path.with_suffix(".wav")
    result_path: Optional[Path] = None

    try:
        async def _synth_segment(seg_text: str, seg_voice: str, dst: Path) -> None:
            for attempt, voice_candidate in enumerate((seg_voice, seg_voice, EDGE_VOICE_RU)):
                try:
                    communicate = edge_tts.Communicate(seg_text, voice=voice_candidate)
                    await communicate.save(str(dst))
                    if dst.exists() and dst.stat().st_size > 0:
                        return
                except Exception as seg_exc:
                    if attempt == 2:
                        raise seg_exc
                    await asyncio.sleep(0.3)

        tasks = []
        for seg_text, seg_voice in segments:
            mp3_fd, mp3_name = tempfile.mkstemp(suffix=".mp3", dir=target_dir)
            os.close(mp3_fd)
            mp3_path = Path(mp3_name)
            part_paths.append(mp3_path)
            tasks.append(_synth_segment(seg_text, seg_voice, mp3_path))

        # Synthesize all Russian and Czech segments in parallel
        await asyncio.gather(*tasks)

        if await asyncio.to_thread(_concat_audio_to_ogg_opus, part_paths, ogg_path):
            result_path = ogg_path
            return result_path
        if part_paths and part_paths[0].exists():
            result_path = part_paths[0]
            return result_path
        return None
    except Exception as exc:
        logger.warning("edge-tts failed (%s), trying Windows SAPI5 offline fallback", exc)
        if await asyncio.to_thread(_synthesize_windows_sapi5, spoken_text, wav_path):
            if await asyncio.to_thread(_concat_audio_to_ogg_opus, [wav_path], ogg_path):
                wav_path.unlink(missing_ok=True)
                result_path = ogg_path
                return result_path
            result_path = wav_path
            return result_path
        wav_path.unlink(missing_ok=True)
        ogg_path.unlink(missing_ok=True)
        return None
    finally:
        for p in part_paths:
            if p != result_path:
                p.unlink(missing_ok=True)
        if result_path != ogg_path:
            ogg_path.unlink(missing_ok=True)


def build_sessions_keyboard(manager: SessionManager) -> InlineKeyboardMarkup:
    active_id = manager.get_active_session().session_id
    buttons = []
    for session in manager.list_sessions():
        prefix = "✅ " if session.session_id == active_id else "💬 "
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{prefix}{session.title}",
                    callback_data=f"switch_session:{session.session_id}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


async def _reject_unauthorized(message: Message) -> None:
    uid = message.from_user.id if message.from_user else None
    await message.answer(
        f"⛔ Доступ закрыт. Ваш Telegram ID: <code>{uid}</code>.\n"
        "Добавьте его в <code>ALLOWED_TELEGRAM_USER_ID</code> в файле <code>.env</code> на ПК.",
        parse_mode="HTML",
    )


@agent_router.message(Command("start", "help"))
async def cmd_start(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return

    import html

    manager = get_session_manager()
    active = manager.get_active_session()
    uid = message.from_user.id if message.from_user else "unknown"
    pin_status = "🔒 Закреплена" if manager.pinned else "🤖 Авто-роутинг"

    await message.answer(
        "👋 Привет! Я Натали, твоя персональная ИИ-ассистентка на домашнем ПК.\n"
        "Отправляй <b>голосовые сообщения</b> или <b>текст</b> — я отвечу текстом и голосом.\n\n"
        f"• Активная сессия: <b>{html.escape(active.title)}</b> ({pin_status})\n"
        f"• Твой Telegram ID: <code>{uid}</code>\n\n"
        "<b>Команды:</b>\n"
        "/sessions — выбрать или посмотреть сессии\n"
        "/new &lt;название&gt; — создать новую сессию\n"
        "/rename &lt;имя&gt; — переименовать текущую сессию\n"
        "/pin / /unpin — закрепить/открепить авто-выбор сессии\n"
        "/clear — очистить историю текущей сессии\n"
        "/voice — вкл/выкл голосовые ответы\n"
        "/harvest &lt;doc_id&gt; — сбор фраз CZECH_HARVESTER в Anki",
        parse_mode="HTML",
    )


@agent_router.message(Command("sessions"))
async def cmd_sessions(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return
    manager = get_session_manager()
    kb = build_sessions_keyboard(manager)
    await message.answer("📂 Выберите сессию:", reply_markup=kb)


@agent_router.callback_query(F.data.startswith("switch_session:"))
async def cb_switch_session(callback: CallbackQuery) -> None:
    if not is_user_allowed(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return
    import html

    session_id = callback.data.split(":", 1)[1]
    manager = get_session_manager()
    switched = manager.switch_session(session_id)
    if switched:
        kb = build_sessions_keyboard(manager)
        await callback.message.edit_text(
            f"📂 Активная сессия: <b>{html.escape(switched.title)}</b>",
            reply_markup=kb,
            parse_mode="HTML",
        )
        await callback.answer(f"Переключено на: {switched.title}")
    else:
        await callback.answer("Сессия не найдена", show_alert=True)


@agent_router.message(Command("new"))
async def cmd_new_session(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return
    import html

    title = (message.text or "").replace("/new", "", 1).strip()
    if not title:
        await message.answer("Укажите название: <code>/new Покупки в интернете</code>", parse_mode="HTML")
        return
    manager = get_session_manager()
    session = manager.create_session(title=title, activate=True)
    await message.answer(f"✅ Создана и активирована сессия: <b>{html.escape(session.title)}</b>", parse_mode="HTML")


@agent_router.message(Command("rename"))
async def cmd_rename_session(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return
    import html

    new_title = (message.text or "").replace("/rename", "", 1).strip()
    if not new_title:
        await message.answer("Укажите новое имя: <code>/rename Новое название</code>", parse_mode="HTML")
        return
    manager = get_session_manager()
    active = manager.get_active_session()
    updated = manager.rename_session(active.session_id, new_title)
    if updated:
        await message.answer(f"✏️ Сессия переименована в: <b>{html.escape(updated.title)}</b>", parse_mode="HTML")


@agent_router.message(Command("pin"))
async def cmd_pin(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return
    import html

    manager = get_session_manager()
    manager.set_pinned(True)
    active = manager.get_active_session()
    await message.answer(
        f"🔒 Сессия <b>{html.escape(active.title)}</b> закреплена (авто-переключение выключено).",
        parse_mode="HTML",
    )


@agent_router.message(Command("unpin"))
async def cmd_unpin(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return
    manager = get_session_manager()
    manager.set_pinned(False)
    await message.answer("🤖 Авто-роутинг сессий снова включён.")


@agent_router.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return
    import html

    manager = get_session_manager()
    cleared = manager.clear_session()
    if cleared:
        await message.answer(f"🧹 История сессии <b>{html.escape(cleared.title)}</b> очищена.", parse_mode="HTML")


@agent_router.message(Command("voice"))
async def cmd_toggle_voice(message: Message) -> None:
    global _voice_replies_enabled
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return
    _voice_replies_enabled = not _voice_replies_enabled
    state = "включены 🔊" if _voice_replies_enabled else "выключены 🔇"
    await message.answer(f"Голосовые ответы {state}.")


async def _process_user_prompt(message: Message, user_text: str) -> None:
    manager = get_session_manager()

    decision = await asyncio.to_thread(manager.route_message, user_text)
    session = manager.sessions[decision.session_id]

    try:
        reply_text = await asyncio.to_thread(manager.generate_reply, user_text, session)
    except Exception as exc:
        logger.exception("Gemini error")
        await message.answer(f"⚠️ Ошибка Gemini: {exc}")
        return

    header = ""
    if decision.created:
        header = f"📂 [Новая сессия: {session.title}]\n\n"
    elif decision.switched:
        header = f"📂 [Сессия: {session.title}]\n\n"

    full_text = f"{header}{reply_text}"
    # Telegram message limit is 4096 chars
    for i in range(0, len(full_text), 4000):
        await message.answer(full_text[i : i + 4000])

    if _voice_replies_enabled:
        try:
            await message.bot.send_chat_action(chat_id=message.chat.id, action="record_voice")
        except Exception:
            pass
        voice_path = await synthesize_voice_file(reply_text)
        if voice_path and voice_path.exists():
            try:
                await message.answer_voice(FSInputFile(voice_path))
            except Exception as exc:
                logger.warning("Failed to send voice reply: %s", exc)
            finally:
                voice_path.unlink(missing_ok=True)


@agent_router.message(F.voice | F.audio)
async def handle_voice_message(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return

    media = message.voice or message.audio
    if not media:
        return

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".ogg", dir=TMP_DIR)
    os.close(fd)
    local_audio = Path(tmp_name)

    try:
        status_msg = await message.answer("🎙 Распознаю голос на ПК...")
        file_info = await message.bot.get_file(media.file_id)
        await message.bot.download_file(file_info.file_path, destination=local_audio)

        transcribed = await asyncio.to_thread(transcribe_voice_file, str(local_audio))
        if not transcribed:
            await status_msg.edit_text("⚠️ Не удалось распознать речь. Попробуйте ещё раз.")
            return

        await status_msg.edit_text(f"🎙 Вы сказали: {transcribed}")
        await _process_user_prompt(message, transcribed)
    except Exception as exc:
        logger.exception("Voice handling failed")
        await message.answer(f"⚠️ Ошибка обработки голоса: {exc}")
    finally:
        local_audio.unlink(missing_ok=True)


@agent_router.message(F.text & ~F.text.startswith("/"))
async def handle_text_message(message: Message) -> None:
    if not is_user_allowed(message.from_user.id if message.from_user else None):
        await _reject_unauthorized(message)
        return

    user_text = (message.text or "").strip()
    if not user_text:
        return

    await _process_user_prompt(message, user_text)
