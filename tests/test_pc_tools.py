import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.pc_tools import (
    _resolve_path,
    get_desktop_path,
    list_antigravity_sessions,
    list_files,
    read_antigravity_session,
    read_text_file,
    run_pc_command,
    write_text_file,
)


class TestPcTools(unittest.TestCase):
    def test_desktop_alias_resolution(self) -> None:
        desktop = Path(get_desktop_path())
        resolved = _resolve_path("desktop/test_note.txt")
        self.assertEqual(resolved, (desktop / "test_note.txt").resolve())

        resolved_ru = _resolve_path("рабочий стол/test_note.txt")
        self.assertEqual(resolved_ru, (desktop / "test_note.txt").resolve())

    def test_write_read_and_list_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "hello.txt"
            res = write_text_file(str(target), "Привет с ПК!")
            self.assertIn("Успешно записан файл", res)
            self.assertEqual(read_text_file(str(target)), "Привет с ПК!")
            listing = list_files(tmp)
            self.assertIn("hello.txt", listing)

    def test_run_pc_command(self) -> None:
        out = run_pc_command("Write-Output 'ok-123'")
        self.assertIn("Exit code: 0", out)
        self.assertIn("ok-123", out)

    def test_antigravity_sessions_list_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brain = Path(tmp)
            logs_dir = brain / "abc-123-def" / ".system_generated" / "logs"
            logs_dir.mkdir(parents=True)
            transcript = logs_dir / "transcript.jsonl"
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "USER_INPUT", "content": "Привет, как дела?"}),
                        json.dumps({"type": "PLANNER_RESPONSE", "content": "Всё отлично!"}),
                    ]
                ),
                encoding="utf-8",
            )
            with patch("src.pc_tools._get_antigravity_brain_dir", return_value=brain):
                listed = list_antigravity_sessions(limit=5)
                self.assertIn("abc-123-def", listed)
                self.assertIn("Привет, как дела?", listed)

                read_out = read_antigravity_session("abc-123")
                self.assertIn("USER: Привет, как дела?", read_out)
                self.assertIn("ASSISTANT: Всё отлично!", read_out)


if __name__ == "__main__":
    unittest.main()
