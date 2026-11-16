from __future__ import annotations

from unittest import mock
import unittest

from src.transcriber import (
    CZECH_LONG_VOWELS,
    HIGHLIGHT_COLOR_RGB,
    generate_vowel_highlight_requests,
    transcribe_audio,
)


class DummySegment:
    def __init__(self, text: str):
        self.text = text


class TestTranscribeAudio(unittest.TestCase):
    @mock.patch("src.transcriber.WhisperModel")
    def test_transcribe_audio_default_call(self, mock_model_cls):
        mock_model = mock.MagicMock()
        mock_model_cls.return_value = mock_model

        # Mock segments generator
        mock_model.transcribe.return_value = (
            [DummySegment(" Dobrý den."), DummySegment(" Jak se dnes máte? ")],
            mock.MagicMock(),
        )

        result = transcribe_audio("test_audio.mp3")

        # Verify model initialization with defaults
        mock_model_cls.assert_called_once_with(
            "large-v3",
            device="cuda",
            compute_type="float16",
        )

        # Verify transcribe call with language="cs"
        mock_model.transcribe.assert_called_once_with(
            "test_audio.mp3",
            language="cs",
        )

        # Verify clean concatenated output
        self.assertEqual(result, "Dobrý den. Jak se dnes máte?")

    @mock.patch("src.transcriber.WhisperModel")
    def test_transcribe_audio_with_empty_and_whitespace_segments(self, mock_model_cls):
        mock_model = mock.MagicMock()
        mock_model_cls.return_value = mock_model

        mock_model.transcribe.return_value = (
            [DummySegment("  "), DummySegment("Jedna věta."), DummySegment("")],
            mock.MagicMock(),
        )

        result = transcribe_audio("audio.wav")
        self.assertEqual(result, "Jedna věta.")

    def test_transcribe_audio_with_custom_model_instance(self):
        mock_custom_model = mock.MagicMock()
        mock_custom_model.transcribe.return_value = (
            [DummySegment("Text z vlastního modelu.")],
            mock.MagicMock(),
        )

        result = transcribe_audio("sample.ogg", model=mock_custom_model)
        self.assertEqual(result, "Text z vlastního modelu.")
        mock_custom_model.transcribe.assert_called_once_with("sample.ogg", language="cs")

    @mock.patch("src.transcriber.WhisperModel")
    def test_transcribe_audio_cuda_fallback_to_cpu(self, mock_model_cls):
        # First call with cuda fails, second with cpu succeeds
        mock_cpu_model = mock.MagicMock()
        mock_cpu_model.transcribe.return_value = (
            [DummySegment("Fallback na CPU funguje.")],
            mock.MagicMock(),
        )
        mock_model_cls.side_effect = [RuntimeError("CUDA device not found"), mock_cpu_model]

        result = transcribe_audio("sample.mp3", device="cuda")
        self.assertEqual(result, "Fallback na CPU funguje.")
        self.assertEqual(mock_model_cls.call_count, 2)
        mock_model_cls.assert_has_calls([
            mock.call("large-v3", device="cuda", compute_type="float16"),
            mock.call("large-v3", device="cpu", compute_type="int8"),
        ])

    def test_transcribe_audio_empty_path_raises_error(self):
        with self.assertRaises(ValueError):
            transcribe_audio("")
        with self.assertRaises(ValueError):
            transcribe_audio("   ")


class TestGenerateVowelHighlightRequests(unittest.TestCase):
    def test_all_lowercase_long_vowels(self):
        text = "á é í ó ú ů ý"
        requests = generate_vowel_highlight_requests(text, document_start_index=1)
        self.assertEqual(len(requests), 7)

        # Expected indices for "á é í ó ú ů ý" (1-based, spaces in between)
        # á at 1, ' ' at 2, é at 3, ' ' at 4, í at 5, ' ' at 6, ó at 7, ' ' at 8, ú at 9, ' ' at 10, ů at 11, ' ' at 12, ý at 13
        expected_starts = [1, 3, 5, 7, 9, 11, 13]
        for req, expected_start in zip(requests, expected_starts):
            range_info = req["updateTextStyle"]["range"]
            self.assertEqual(range_info["startIndex"], expected_start)
            self.assertEqual(range_info["endIndex"], expected_start + 1)
            # Verify styling
            style = req["updateTextStyle"]["textStyle"]
            self.assertTrue(style["bold"])
            self.assertEqual(
                style["foregroundColor"]["color"]["rgbColor"],
                HIGHLIGHT_COLOR_RGB,
            )
            self.assertEqual(req["updateTextStyle"]["fields"], "bold,foregroundColor")

    def test_all_uppercase_long_vowels(self):
        text = "Á É Í Ó Ú Ů Ý"
        requests = generate_vowel_highlight_requests(text, document_start_index=10)
        self.assertEqual(len(requests), 7)
        expected_starts = [10, 12, 14, 16, 18, 20, 22]
        for req, expected_start in zip(requests, expected_starts):
            range_info = req["updateTextStyle"]["range"]
            self.assertEqual(range_info["startIndex"], expected_start)
            self.assertEqual(range_info["endIndex"], expected_start + 1)

    def test_short_vowels_and_consonants_ignored(self):
        # Note: 'ě' is a short vowel with caron in Czech and must NOT be highlighted
        text = "a e i o u ě y b c d f g h ch k l m n p r s t v z č š ž ř ď ť ň"
        requests = generate_vowel_highlight_requests(text)
        self.assertEqual(len(requests), 0)

    def test_real_czech_sentence_mixed_vowels(self):
        # Sentence: "Můj milý přítel má novou knihu."
        # Long vowels:
        # - 'ů' at offset 1
        # - 'ý' at offset 7
        # - 'í' at offset 11
        # - 'á' at offset 17
        text = "Můj milý přítel má novou knihu."
        doc_start = 5
        requests = generate_vowel_highlight_requests(text, document_start_index=doc_start)
        self.assertEqual(len(requests), 4)

        # 'ů' at 5 + 1 = 6
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["startIndex"], doc_start + 1)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["endIndex"], doc_start + 2)

        # 'ý' at 5 + 7 = 12
        self.assertEqual(requests[1]["updateTextStyle"]["range"]["startIndex"], doc_start + 7)
        self.assertEqual(requests[1]["updateTextStyle"]["range"]["endIndex"], doc_start + 8)

        # 'í' at 5 + 11 = 16
        self.assertEqual(requests[2]["updateTextStyle"]["range"]["startIndex"], doc_start + 11)
        self.assertEqual(requests[2]["updateTextStyle"]["range"]["endIndex"], doc_start + 12)

        # 'á' at 5 + 17 = 22
        self.assertEqual(requests[3]["updateTextStyle"]["range"]["startIndex"], doc_start + 17)
        self.assertEqual(requests[3]["updateTextStyle"]["range"]["endIndex"], doc_start + 18)

    def test_utf16_surrogate_pairs_emojis(self):
        """
        Tests that characters outside BMP (e.g. emojis with code points > 0xFFFF)
        which consume 2 UTF-16 code units properly shift subsequent vowel indices.
        """
        # Emoji '🎧' (U+1F3A7) is 2 UTF-16 code units.
        # '🎧' (2 units) + ' ' (1 unit) + 'd' (1 unit) + 'r' (1 unit) + 'a' (1 unit) + 'h' (1 unit) + 'á' (1 unit)
        # 'á' starts at UTF-16 offset: 2 + 1 + 1 + 1 + 1 + 1 = 7.
        # If document_start_index = 1, start_idx must be 1 + 7 = 8.
        # (In standard Python string indexing, 'á' is at char index 6, which would give 7!).
        text = "🎧 drahá"
        requests = generate_vowel_highlight_requests(text, document_start_index=1)
        self.assertEqual(len(requests), 1)

        req_range = requests[0]["updateTextStyle"]["range"]
        self.assertEqual(req_range["startIndex"], 8)
        self.assertEqual(req_range["endIndex"], 9)

    def test_multiple_emojis_before_vowel(self):
        # 3 astral-plane emojis: 🎉 (U+1F389), 🚀 (U+1F680), 💡 (U+1F4A1)
        # Each consumes 2 UTF-16 code units -> 6 units.
        # + 1 space = 7 units.
        # + 'k' (1 unit) + 'r' (1 unit) = 9 units before 'á'.
        # With doc_start = 1, 'á' starts at 1 + 9 = 10, ends at 11.
        text = "\U0001f389\U0001f680\U0001f4a1 král"
        requests = generate_vowel_highlight_requests(text, document_start_index=1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["startIndex"], 10)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["endIndex"], 11)

    def test_bmp_symbol_vs_astral_plane_emoji(self):
        # ✨ (U+2728, sparkles) is in the BMP (ord <= 0xFFFF) -> 1 UTF-16 code unit.
        # 🚀 (U+1F680, rocket) is in Astral Plane (ord > 0xFFFF) -> 2 UTF-16 code units.
        # Text: "✨🚀 král" -> 1 + 2 + 1(space) + 2('kr') = 6 units before 'á'.
        # With doc_start = 1, 'á' starts at 1 + 6 = 7.
        text = "\u2728\U0001f680 král"
        requests = generate_vowel_highlight_requests(text, document_start_index=1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["startIndex"], 7)
        self.assertEqual(requests[0]["updateTextStyle"]["range"]["endIndex"], 8)

    def test_empty_string(self):
        self.assertEqual(generate_vowel_highlight_requests(""), [])

    def test_none_handling(self):
        self.assertEqual(generate_vowel_highlight_requests(None), [])  # type: ignore


if __name__ == "__main__":
    unittest.main()
