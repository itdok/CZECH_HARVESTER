from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import List


def _resolve_path(raw_path: str) -> Path:
    """
    Resolves user-friendly path aliases like 'desktop', 'рабочий стол', or '~'
    into an absolute Path on Windows.
    """
    cleaned = raw_path.strip().strip('"').strip("'")
    lowered = cleaned.lower().replace("\\", "/")

    desktop = Path(get_desktop_path())
    for prefix in ("desktop/", "рабочий стол/"):
        if lowered.startswith(prefix):
            rel = cleaned[len(prefix) :]
            return (desktop / rel).resolve()

    if lowered in ("desktop", "рабочий стол"):
        return desktop

    p = Path(cleaned).expanduser()
    if not p.is_absolute():
        # Default relative paths without directory prefix that look like desktop requests
        # stay relative to workspace unless Desktop was explicitly requested
        p = Path.cwd() / p
    return p.resolve()


def get_desktop_path() -> str:
    """Returns the full path to the current user's Windows Desktop directory."""
    user_profile = Path(os.path.expanduser("~"))
    onedrive_desktop = user_profile / "OneDrive" / "Desktop"
    std_desktop = user_profile / "Desktop"
    if std_desktop.exists():
        return str(std_desktop)
    if onedrive_desktop.exists():
        return str(onedrive_desktop)
    return str(std_desktop)


def list_files(directory: str = ".") -> str:
    """
    Lists files and subdirectories in the given directory.
    Supports 'desktop', 'рабочий стол', '.', or any full/relative path.
    """
    target = _resolve_path(directory)
    if not target.exists():
        return f"Ошибка: папка не найдена: {target}"
    if not target.is_dir():
        return f"Ошибка: это не папка: {target}"

    entries: List[str] = []
    for item in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:100]:
        kind = "[DIR]" if item.is_dir() else "[FILE]"
        entries.append(f"{kind} {item.name}")

    if not entries:
        return f"Папка пуста: {target}"
    return f"Содержимое {target}:\n" + "\n".join(entries)


def read_text_file(path: str) -> str:
    """
    Reads and returns the contents of a text file on the PC (up to 15000 characters).
    Supports 'desktop/file.txt', 'рабочий стол/file.txt', or any path.
    """
    target = _resolve_path(path)
    if not target.exists():
        return f"Ошибка: файл не найден: {target}"
    if not target.is_file():
        return f"Ошибка: это не файл: {target}"

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > 15000:
            return content[:15000] + "\n... [обрезано]"
        return content
    except Exception as exc:
        return f"Ошибка чтения файла {target}: {exc}"


def write_text_file(path: str, content: str, overwrite: bool = True) -> str:
    """
    Creates or writes a text file on the PC.
    Supports 'desktop/note.txt', 'рабочий стол/note.txt', or any full/relative path.
    """
    target = _resolve_path(path)
    if target.exists() and not overwrite:
        return f"Файл уже существует: {target}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Успешно записан файл: {target} ({len(content)} симв.)"
    except Exception as exc:
        return f"Ошибка записи файла {target}: {exc}"


def run_pc_command(command: str) -> str:
    """
    Executes a PowerShell command on the user's Windows PC and returns stdout/stderr.
    Use for running scripts, checking git status, launching tests, etc.
    """
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        parts = [f"Exit code: {proc.returncode}"]
        if output:
            parts.append(f"STDOUT:\n{output[:4000]}")
        if err:
            parts.append(f"STDERR:\n{err[:2000]}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Ошибка: команда превысила таймаут 30 секунд."
    except Exception as exc:
        return f"Ошибка выполнения команды: {exc}"


def _get_antigravity_brain_dir() -> Path:
    return Path.home() / ".gemini" / "antigravity" / "brain"


def list_antigravity_sessions(limit: int = 10) -> str:
    """
    Lists recent Antigravity IDE/desktop conversation sessions stored locally on this PC.
    Returns session IDs, modification times, and initial user prompts.
    """
    brain_dir = _get_antigravity_brain_dir()
    if not brain_dir.exists():
        return f"Папка сессий Antigravity не найдена: {brain_dir}"

    transcripts: List[Path] = []
    for sub in brain_dir.iterdir():
        if sub.is_dir():
            t = sub / ".system_generated" / "logs" / "transcript.jsonl"
            if t.exists():
                transcripts.append(t)

    if not transcripts:
        return "Локальные транскрипты сессий Antigravity не найдены."

    transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    lines: List[str] = []
    for t_path in transcripts[: max(1, limit)]:
        conv_id = t_path.parent.parent.parent.name
        first_prompt = "(без текста)"
        try:
            with t_path.open("r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    if not raw_line.strip():
                        continue
                    obj = json.loads(raw_line)
                    if obj.get("type") == "USER_INPUT" and obj.get("content"):
                        first_prompt = str(obj["content"]).replace("\n", " ").strip()[:100]
                        break
        except Exception:
            pass
        lines.append(f"- {conv_id}: {first_prompt}")

    return "Последние сессии Antigravity на ПК:\n" + "\n".join(lines)


def read_antigravity_session(session_id: str, last_messages: int = 20) -> str:
    """
    Reads recent user/assistant messages from a local Antigravity session transcript on this PC.
    Supports full UUID or unique prefix of session_id.
    """
    brain_dir = _get_antigravity_brain_dir()
    if not brain_dir.exists():
        return f"Папка сессий Antigravity не найдена: {brain_dir}"

    target_file: Path | None = None
    cleaned_id = session_id.strip()
    for sub in brain_dir.iterdir():
        if sub.is_dir() and (sub.name == cleaned_id or sub.name.startswith(cleaned_id)):
            candidate = sub / ".system_generated" / "logs" / "transcript.jsonl"
            if candidate.exists():
                target_file = candidate
                break

    if target_file is None:
        return f"Сессия Antigravity с ID '{session_id}' не найдена."

    messages: List[str] = []
    try:
        with target_file.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                obj = json.loads(raw_line)
                step_type = obj.get("type")
                content = (obj.get("content") or "").strip()
                if not content:
                    continue
                if step_type == "USER_INPUT":
                    messages.append(f"USER: {content[:600]}")
                elif step_type == "PLANNER_RESPONSE":
                    messages.append(f"ASSISTANT: {content[:800]}")
    except Exception as exc:
        return f"Ошибка чтения транскрипта {target_file}: {exc}"

    if not messages:
        return "В этой сессии пока нет текстовых сообщений."

    tail = messages[-max(1, last_messages) :]
    joined = "\n\n".join(tail)
    if len(joined) > 14000:
        joined = joined[-14000:]
    return f"История сессии {target_file.parent.parent.parent.name} (последние {len(tail)} реплик):\n\n{joined}"


PC_TOOLS = [
    get_desktop_path,
    list_files,
    read_text_file,
    write_text_file,
    run_pc_command,
    list_antigravity_sessions,
    read_antigravity_session,
]
