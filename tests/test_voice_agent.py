from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.voice_agent import (
    EDGE_VOICE_CS,
    EDGE_VOICE_EN,
    EDGE_VOICE_RU,
    clean_text_for_tts,
    detect_tts_voice,
    is_user_allowed,
    split_bilingual_segments,
    split_voice_chunks,
    transcribe_voice_file,
)


class TestVoiceAgent(unittest.TestCase):
    def test_detect_tts_voice(self) -> None:
        self.assertEqual(detect_tts_voice("Привет, как дела?"), EDGE_VOICE_RU)
        self.assertEqual(detect_tts_voice("Dobrý den, jak se máte? Příliš žluťoučký kůň."), EDGE_VOICE_CS)
        self.assertEqual(detect_tts_voice("Hello, how are you today?"), EDGE_VOICE_RU)

    def test_clean_text_for_tts(self) -> None:
        raw = "**Привет!** Вот [ссылка](https://example.com) и код:\n```python\nprint(1)\n```"
        cleaned = clean_text_for_tts(raw)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("```", cleaned)
        self.assertIn("ссылка", cleaned)
        self.assertIn("код", cleaned)

    def test_is_user_allowed(self) -> None:
        with patch.dict("os.environ", {"ALLOWED_TELEGRAM_USER_ID": ""}):
            self.assertTrue(is_user_allowed(12345))

        with patch.dict("os.environ", {"ALLOWED_TELEGRAM_USER_ID": "111, 222"}):
            self.assertTrue(is_user_allowed(111))
            self.assertTrue(is_user_allowed(222))
            self.assertFalse(is_user_allowed(999))
            self.assertFalse(is_user_allowed(None))

    def test_transcribe_voice_file_uses_auto_language(self) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [SimpleNamespace(text=" Привет "), SimpleNamespace(text="как дела ")],
            SimpleNamespace(language="ru"),
        )
        result = transcribe_voice_file("dummy.ogg", model=mock_model)
        self.assertEqual(result, "Привет как дела")
        mock_model.transcribe.assert_called_once_with("dummy.ogg", language=None, vad_filter=True, beam_size=1)

    def test_split_bilingual_segments(self) -> None:
        text = "По-чешски это звучит так: Dobrý den, jak se máte? Очень просто!"
        segments = split_bilingual_segments(text)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0][1], EDGE_VOICE_RU)
        self.assertIn("По-чешски", segments[0][0])
        self.assertEqual(segments[1][1], EDGE_VOICE_CS)
        self.assertIn("Dobrý den", segments[1][0])
        self.assertEqual(segments[2][1], EDGE_VOICE_RU)
        self.assertIn("Очень просто", segments[2][0])

        tech_text = "Я создала файл test.txt через Gemini в Telegram."
        tech_segments = split_bilingual_segments(tech_text)
        self.assertEqual(len(tech_segments), 1)
        self.assertEqual(tech_segments[0][1], EDGE_VOICE_RU)

    def test_split_voice_chunks(self) -> None:
        short = "Короткий ответ."
        self.assertEqual(split_voice_chunks(short), [short])

        long_text = (
            "Первое предложение достаточно длинное для теста. "
            "Второе предложение тоже добавляет объёма, чтобы перейти порог разбиения. "
            "Третье предложение продолжает мысль и уходит во вторую часть ответа, чтобы убедиться, "
            "что разбиение происходит аккуратно по границе предложения и ничего не теряется."
        )
        chunks = split_voice_chunks(long_text, first_chunk_max=100)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("."))


if __name__ == "__main__":
    unittest.main()
