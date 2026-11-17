from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.pc_tools import PC_TOOLS

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SESSIONS_DIR = BASE_DIR / "data" / "sessions"
DEFAULT_MODEL = os.getenv("GEMINI_AGENT_MODEL", "gemini-3-flash-preview")
ROUTER_MODEL = os.getenv("GEMINI_ROUTER_MODEL", "gemini-flash-lite-latest")
FALLBACK_MODEL = "gemini-2.5-flash"
MAX_HISTORY_TURNS = 24

DEFAULT_CZECH_SESSION_ID = "czech_learning"
DEFAULT_CZECH_SYSTEM_PROMPT = (
    "Ты — Натали (Nathalee), персональная ИИ-ассистентка Gemini, работающая локально на Windows-ПК пользователя "
    "в проекте CZECH_HARVESTER (и репетитор чешского языка). Всегда отвечай от женского лица "
    "(например: 'я поняла', 'я создала', 'я проверила', 'я нашла'). "
    "У тебя ЕСТЬ инструменты для работы с ПК: создание и чтение файлов (включая рабочий стол через 'desktop/...'), "
    "просмотр папок, выполнение PowerShell-команд, а также чтение локальных сессий Antigravity "
    "(list_antigravity_sessions / read_antigravity_session). Когда пользователь просит создать файл, проверить проект, "
    "выполнить действие на ПК или посмотреть другой диалог Antigravity — обязательно используй свои инструменты. "
    "Отвечай кратко, по делу и дружелюбно (на русском, с примерами на чешском, если речь о языке). "
    "Избегай сложных таблиц, чтобы ответ хорошо звучал при озвучке голосом."
)
DEFAULT_GENERAL_SYSTEM_PROMPT = (
    "Ты — Натали (Nathalee), персональная ИИ-ассистентка Gemini, работающая локально на Windows-ПК пользователя. "
    "Всегда отвечай от женского лица (например: 'я поняла', 'я сделала', 'я проверила'). "
    "У тебя ЕСТЬ инструменты для работы с файлами (в том числе на рабочем столе через 'desktop/...'), "
    "просмотра папок, выполнения PowerShell-команд и чтения локальных сессий Antigravity "
    "(list_antigravity_sessions / read_antigravity_session). Когда пользователь просит создать/прочитать файл, "
    "сделать что-то на ПК или подхватить контекст из другой сессии — используй инструменты, а не отказывайся. "
    "Отвечай кратко, чётко и по делу, без громоздких таблиц, так как ответы озвучиваются голосом в Telegram."
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    if not cleaned:
        cleaned = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return cleaned[:40]


@dataclass
class Session:
    """Persistent conversation session stored on disk."""

    session_id: str
    title: str
    description: str
    system_prompt: str
    history: List[Dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def add_message(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text, "ts": _utc_now_iso()})
        self.updated_at = _utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=str(data["session_id"]),
            title=str(data.get("title", data["session_id"])),
            description=str(data.get("description", "")),
            system_prompt=str(data.get("system_prompt", DEFAULT_GENERAL_SYSTEM_PROMPT)),
            history=list(data.get("history", [])),
            created_at=str(data.get("created_at", _utc_now_iso())),
            updated_at=str(data.get("updated_at", _utc_now_iso())),
        )


@dataclass
class RouteDecision:
    """Result of automatic session routing."""

    action: str  # "continue" | "switch" | "create"
    session_id: str
    title: str
    switched: bool = False
    created: bool = False
    reason: str = ""


class SessionManager:
    """
    Manages persistent conversation sessions on disk and routes incoming
    messages to the appropriate session using Gemini.
    """

    def __init__(
        self,
        sessions_dir: Optional[Path] = None,
        client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.sessions_dir = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.sessions_dir / "state.json"
        self.model = model
        self._client = client

        self.sessions: Dict[str, Session] = {}
        self.active_session_id: str = DEFAULT_CZECH_SESSION_ID
        self.pinned: bool = False

        self._load_all()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment/.env")
        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )
        return self._client

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _save_session(self, session: Session) -> None:
        path = self._session_path(session.session_id)
        path.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_state(self) -> None:
        payload = {
            "active_session_id": self.active_session_id,
            "pinned": self.pinned,
            "updated_at": _utc_now_iso(),
        }
        self.state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_all(self) -> None:
        for file_path in sorted(self.sessions_dir.glob("*.json")):
            if file_path.name == "state.json":
                continue
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                session = Session.from_dict(data)
                self.sessions[session.session_id] = session
            except Exception:
                continue

        if DEFAULT_CZECH_SESSION_ID not in self.sessions:
            default_session = Session(
                session_id=DEFAULT_CZECH_SESSION_ID,
                title="Чешский язык (CZECH_HARVESTER)",
                description="Изучение чешского языка, грамматика, перевод фраз, карточки Anki и проект CZECH_HARVESTER.",
                system_prompt=DEFAULT_CZECH_SYSTEM_PROMPT,
            )
            self.sessions[default_session.session_id] = default_session
            self._save_session(default_session)

        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                loaded_id = state.get("active_session_id", DEFAULT_CZECH_SESSION_ID)
                if loaded_id in self.sessions:
                    self.active_session_id = loaded_id
                self.pinned = bool(state.get("pinned", False))
            except Exception:
                pass

        if self.active_session_id not in self.sessions:
            self.active_session_id = next(iter(self.sessions))
        self._save_state()

    def get_active_session(self) -> Session:
        if self.active_session_id not in self.sessions:
            self.active_session_id = next(iter(self.sessions))
            self._save_state()
        return self.sessions[self.active_session_id]

    def list_sessions(self) -> List[Session]:
        return sorted(
            self.sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )

    def create_session(
        self,
        title: str,
        description: str = "",
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        activate: bool = True,
    ) -> Session:
        base_id = _slugify(session_id or title)
        unique_id = base_id
        counter = 2
        while unique_id in self.sessions:
            unique_id = f"{base_id}_{counter}"
            counter += 1

        session = Session(
            session_id=unique_id,
            title=title.strip() or unique_id,
            description=description.strip() or f"Диалог на тему: {title.strip()}",
            system_prompt=system_prompt or DEFAULT_GENERAL_SYSTEM_PROMPT,
        )
        self.sessions[unique_id] = session
        self._save_session(session)

        if activate:
            self.active_session_id = unique_id
            self._save_state()

        return session

    def switch_session(self, session_id: str) -> Optional[Session]:
        if session_id not in self.sessions:
            return None
        self.active_session_id = session_id
        self._save_state()
        return self.sessions[session_id]

    def rename_session(self, session_id: str, new_title: str) -> Optional[Session]:
        session = self.sessions.get(session_id)
        if not session or not new_title.strip():
            return None
        session.title = new_title.strip()
        session.updated_at = _utc_now_iso()
        self._save_session(session)
        return session

    def clear_session(self, session_id: Optional[str] = None) -> Optional[Session]:
        target_id = session_id or self.active_session_id
        session = self.sessions.get(target_id)
        if not session:
            return None
        session.history.clear()
        session.updated_at = _utc_now_iso()
        self._save_session(session)
        return session

    def set_pinned(self, pinned: bool) -> None:
        self.pinned = pinned
        self._save_state()

    def route_message(self, user_message: str) -> RouteDecision:
        """
        Decides whether the incoming message belongs to the active session,
        should switch to an existing session, or requires creating a new session.
        """
        active = self.get_active_session()
        if self.pinned:
            return RouteDecision(
                action="continue",
                session_id=active.session_id,
                title=active.title,
                reason="pinned",
            )

        if len(self.sessions) == 1 and len(user_message.strip()) < 15:
            return RouteDecision(
                action="continue",
                session_id=active.session_id,
                title=active.title,
                reason="short_single_session",
            )

        sessions_summary = [
            {
                "session_id": s.session_id,
                "title": s.title,
                "description": s.description,
                "is_active": s.session_id == active.session_id,
            }
            for s in self.list_sessions()
        ]
        recent_turns = active.history[-4:] if active.history else []
        recent_text = "\n".join(
            f"{m['role']}: {m['text'][:160]}" for m in recent_turns
        ) or "(история пуста)"

        router_prompt = (
            "Ты — маршрутизатор диалоговых сессий персонального ассистента.\n"
            "Определи, к какой сессии относится новое сообщение пользователя.\n\n"
            f"Текущая активная сессия: {active.session_id} ({active.title})\n"
            f"Последние реплики в активной сессии:\n{recent_text}\n\n"
            f"Все существующие сессии:\n{json.dumps(sessions_summary, ensure_ascii=False)}\n\n"
            f"Новое сообщение пользователя:\n{user_message}\n\n"
            "Правила:\n"
            "1. Если сообщение продолжает текущую тему, является уточнением ('а почему?', 'переведи', 'ещё') "
            "или приветствием — верни action='continue' и target_session_id текущей сессии.\n"
            "2. Если сообщение явно относится к другой УЖЕ существующей сессии — верни action='switch' и её target_session_id.\n"
            "3. Если сообщение начинает принципиально новую тему, для которой нет подходящей сессии "
            "(например, пользователь обсуждал чешский язык, а теперь спрашивает про покупки в интернете, "
            "здоровье, путешествия или программирование) — верни action='create', задай короткое понятное "
            "new_title (2-4 слова на русском) и new_description (1 предложение).\n\n"
            "Верни ТОЛЬКО валидный JSON:\n"
            '{"action": "continue|switch|create", "target_session_id": "...", '
            '"new_title": "...", "new_description": "..."}'
        )

        try:
            client = self._get_client()
            resp = client.models.generate_content(
                model=ROUTER_MODEL,
                contents=router_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            raw = (resp.text or "").strip()
            decision_data = json.loads(raw)
        except Exception:
            return RouteDecision(
                action="continue",
                session_id=active.session_id,
                title=active.title,
                reason="router_fallback",
            )

        action = str(decision_data.get("action", "continue")).lower()
        target_id = str(decision_data.get("target_session_id", active.session_id))

        if action == "switch" and target_id in self.sessions and target_id != active.session_id:
            switched = self.switch_session(target_id)
            if switched:
                return RouteDecision(
                    action="switch",
                    session_id=switched.session_id,
                    title=switched.title,
                    switched=True,
                )

        if action == "create":
            new_title = str(decision_data.get("new_title", "")).strip() or "Новая тема"
            new_desc = str(decision_data.get("new_description", "")).strip() or new_title
            created = self.create_session(
                title=new_title,
                description=new_desc,
                system_prompt=(
                    f"{DEFAULT_GENERAL_SYSTEM_PROMPT}\n"
                    f"Текущая тема сессии: {new_title} — {new_desc}"
                ),
                activate=True,
            )
            return RouteDecision(
                action="create",
                session_id=created.session_id,
                title=created.title,
                created=True,
            )

        return RouteDecision(
            action="continue",
            session_id=active.session_id,
            title=active.title,
        )

    def generate_reply(
        self,
        user_message: str,
        session: Optional[Session] = None,
    ) -> str:
        """
        Generates a Gemini response using the session's system prompt and history,
        persisting both turns to disk.
        """
        target = session or self.get_active_session()
        client = self._get_client()

        contents: List[types.Content] = []
        for turn in target.history[-MAX_HISTORY_TURNS:]:
            role = "user" if turn.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn.get("text", ""))],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )
        )

        config = types.GenerateContentConfig(
            system_instruction=target.system_prompt,
            tools=PC_TOOLS,
            temperature=0.6,
        )

        reply_text = ""
        last_error: Optional[Exception] = None
        candidates = []
        for m in (self.model, ROUTER_MODEL, "gemini-3.1-flash-lite-preview", FALLBACK_MODEL, "gemini-flash-latest"):
            if m not in candidates:
                candidates.append(m)

        for model_name in candidates:
            for attempt in range(2):
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    reply_text = (resp.text or "").strip()
                    if reply_text:
                        break
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc)
                    if "503" in err_str and attempt == 0:
                        time.sleep(1.5)
                        continue
                    if "429" in err_str or "503" in err_str:
                        time.sleep(1.0)
                    break
            if reply_text:
                break

        if not reply_text:
            raise RuntimeError(f"Gemini generation failed: {last_error}")

        target.add_message("user", user_message)
        target.add_message("model", reply_text)
        self._save_session(target)
        self._save_state()

        return reply_text
