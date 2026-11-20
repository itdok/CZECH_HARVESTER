from __future__ import annotations

import json
from unittest import mock
import unittest

from pydantic import ValidationError

from src.card_generator import (
    CardBatch,
    CardGenerationError,
    CardItem,
    _build_prompt,
    _ensure_bold_phrase,
    _parse_card_batch_response,
    format_cloze_sentence,
    generate_cards,
    strip_cloze_and_html,
)


class TestCardModels(unittest.TestCase):
    def test_card_item_valid(self):
        item = CardItem(
            f="To je <b>drahá kniha</b>.",
            t="Это дорогая книга.",
            g="drahý (příd. jm., tvrdé vzor mladý), kniha (podst. jm., vzor žena)",
        )
        self.assertEqual(item.f, "To je <b>drahá kniha</b>.")
        self.assertEqual(item.t, "Это дорогая книга.")
        self.assertIn("drahý", item.g)

    def test_card_item_missing_fields_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            CardItem(f="Jen front")  # missing t and g

    def test_card_batch_valid(self):
        batch = CardBatch(
            cards=[
                CardItem(f="<b>Ahoj</b>!", t="Привет!", g="междометие"),
                CardItem(f="<b>Dobrý den</b>.", t="Добрый день.", g="вежливое приветствие"),
            ]
        )
        self.assertEqual(len(batch.cards), 2)
        dumped = batch.model_dump()
        self.assertIn("cards", dumped)
        self.assertEqual(len(dumped["cards"]), 2)


class TestEnsureBoldPhrase(unittest.TestCase):
    def test_already_has_bold_tags(self):
        text = "To je <b>drahá kniha</b>."
        result = _ensure_bold_phrase(text, "drahá kniha")
        self.assertEqual(result, "To je <b>drahá kniha</b>.")

    def test_wraps_target_phrase(self):
        text = "To je drahá kniha."
        result = _ensure_bold_phrase(text, "drahá kniha")
        self.assertEqual(result, "To je <b>drahá kniha</b>.")

    def test_preserves_case_when_wrapping(self):
        text = "Drahá kniha leží na stole."
        result = _ensure_bold_phrase(text, "drahá kniha")
        self.assertEqual(result, "<b>Drahá kniha</b> leží na stole.")

    def test_phrase_not_found_in_text(self):
        text = "Nějaký kontext bez cílového slova."
        result = _ensure_bold_phrase(text, "včera večer")
        self.assertEqual(result, "Nějaký kontext bez cílového slova. (<b>včera večer</b>)")

    def test_empty_text(self):
        self.assertEqual(_ensure_bold_phrase("", "slovo"), "<b>slovo</b>")
        self.assertEqual(_ensure_bold_phrase("", ""), "")


class TestPromptBuilding(unittest.TestCase):
    def test_build_prompt_with_context(self):
        phrases = [
            {"phrase": "mít se dobře", "context": "Doufám, že se máš dobře."},
            {"phrase": "drahá kniha", "context": "To je drahá kniha."},
        ]
        prompt = _build_prompt(phrases)
        self.assertIn("1. Target: 'mít se dobře'", prompt)
        self.assertIn("Doufám, že se máš dobře.", prompt)
        self.assertIn("2. Target: 'drahá kniha'", prompt)

    def test_build_prompt_without_context(self):
        phrases = [{"phrase": "osamělé slovo"}]
        prompt = _build_prompt(phrases)
        self.assertIn("Target: 'osamělé slovo'", prompt)
        self.assertIn("Full context: 'osamělé slovo'", prompt)


class TestGenerateCardsGemini(unittest.TestCase):
    def test_empty_or_whitespace_phrases_return_empty_list(self):
        self.assertEqual(generate_cards([]), [])
        self.assertEqual(generate_cards([{"phrase": "   "}]), [])
        self.assertEqual(generate_cards([{"other": "key"}]), [])

    @mock.patch("src.card_generator.genai.Client")
    def test_generate_cards_gemini_with_parsed_response(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        batch_obj = CardBatch(
            cards=[
                CardItem(
                    f="To je <b>drahá kniha</b>.",
                    t="Это дорогая книга.",
                    g="drahý - прил., kniha - сущ., ж.р.",
                )
            ]
        )
        mock_response = mock.MagicMock()
        mock_response.parsed = batch_obj
        mock_client.models.generate_content.return_value = mock_response

        phrases = [{"phrase": "drahá kniha", "context": "To je drahá kniha."}]
        cards = generate_cards(phrases, api_key="fake-test-key")

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["f"], "To je <b>drahá kniha</b>.")
        self.assertEqual(cards[0]["t"], "Это дорогая книга.")
        self.assertIn("drahý", cards[0]["g"])
        mock_client.models.generate_content.assert_called_once()

    @mock.patch("src.card_generator.genai.Client")
    def test_generate_cards_gemini_with_text_json_response(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        json_payload = json.dumps({
            "cards": [
                {
                    "f": "Mám se <b>dobře</b>.",
                    "t": "У меня все хорошо.",
                    "g": "dobře - наречие",
                }
            ]
        })
        mock_response = mock.MagicMock()
        mock_response.parsed = None
        mock_response.text = json_payload
        mock_client.models.generate_content.return_value = mock_response

        phrases = [{"phrase": "dobře", "context": "Mám se dobře."}]
        cards = generate_cards(phrases, api_key="fake-test-key")

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["f"], "Mám se <b>dobře</b>.")
        self.assertEqual(cards[0]["t"], "У меня все хорошо.")

    @mock.patch("src.card_generator.genai.Client")
    def test_generate_cards_gemini_ensures_bold_if_model_forgot(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client

        batch_obj = CardBatch(
            cards=[
                CardItem(
                    f="To je drahá kniha.",  # model forgot <b>...</b>
                    t="Это дорогая книга.",
                    g="gramatika",
                )
            ]
        )
        mock_response = mock.MagicMock()
        mock_response.parsed = batch_obj
        mock_client.models.generate_content.return_value = mock_response

        phrases = [{"phrase": "drahá kniha", "context": "To je drahá kniha."}]
        cards = generate_cards(phrases, api_key="fake-test-key")

        self.assertEqual(cards[0]["f"], "To je <b>drahá kniha</b>.")

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error(self):
        phrases = [{"phrase": "slovo"}]
        with self.assertRaises(CardGenerationError) as ctx:
            generate_cards(phrases, api_key=None, client=None)
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    @mock.patch("src.card_generator.genai.Client")
    def test_gemini_api_call_failure_raises_card_generation_error(self, mock_client_cls):
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = RuntimeError("API Quota Exceeded")

        phrases = [{"phrase": "slovo"}]
        with self.assertRaises(CardGenerationError) as ctx:
            generate_cards(phrases, api_key="fake-key")
        self.assertIn("Gemini API generation failed", str(ctx.exception))


class TestGenerateCardsOllama(unittest.TestCase):
    @mock.patch("src.card_generator.requests.post")
    def test_ollama_generate_endpoint_success(self, mock_post):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": json.dumps({
                "cards": [
                    {
                        "f": "Bydlím v <b>Praze</b>.",
                        "t": "Я живу в Праге.",
                        "g": "Praha - 6. pád (lokál) s předložkou v",
                    }
                ]
            })
        }
        mock_post.return_value = mock_resp

        phrases = [{"phrase": "Praze", "context": "Bydlím v Praze."}]
        cards = generate_cards(
            phrases,
            use_local_ollama=True,
            ollama_endpoint="http://localhost:11434/api/generate",
            ollama_model="qwen2.5:7b",
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["f"], "Bydlím v <b>Praze</b>.")
        self.assertEqual(cards[0]["t"], "Я живу в Праге.")
        self.assertIn("lokál", cards[0]["g"])

        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        self.assertEqual(call_args[0], "http://localhost:11434/api/generate")
        self.assertEqual(call_kwargs["json"]["model"], "qwen2.5:7b")
        self.assertFalse(call_kwargs["json"]["stream"])

    @mock.patch("src.card_generator.requests.post")
    def test_ollama_chat_endpoint_format(self, mock_post):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "cards": [
                        {
                            "f": "Mám <b>čas</b>.",
                            "t": "У меня есть время.",
                            "g": "čas - podst. jm. muž. neživotné, 4. pád",
                        }
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        phrases = [{"phrase": "čas", "context": "Mám čas."}]
        cards = generate_cards(phrases, use_local_ollama=True)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["f"], "Mám <b>čas</b>.")

    @mock.patch("src.card_generator.requests.post")
    def test_ollama_network_error_raises_card_generation_error(self, mock_post):
        mock_post.side_effect = ConnectionError("Could not reach Ollama at localhost:11434")

        phrases = [{"phrase": "test"}]
        with self.assertRaises(CardGenerationError) as ctx:
            generate_cards(phrases, use_local_ollama=True)
        self.assertIn("Ollama local router request failed", str(ctx.exception))

    @mock.patch("src.card_generator.requests.post")
    def test_ollama_invalid_json_raises_card_generation_error(self, mock_post):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Not a JSON content"}
        mock_post.return_value = mock_resp

        phrases = [{"phrase": "test"}]
        with self.assertRaises(CardGenerationError) as ctx:
            generate_cards(phrases, use_local_ollama=True)
        self.assertIn("Failed to validate Ollama response", str(ctx.exception))


class TestFallbackRouting(unittest.TestCase):
    @mock.patch("src.card_generator.requests.post")
    @mock.patch("src.card_generator.genai.Client")
    def test_fallback_to_ollama_when_gemini_fails(self, mock_client_cls, mock_post):
        # Gemini fails
        mock_client = mock.MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = RuntimeError("Gemini 503 Overloaded")

        # Ollama succeeds
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": json.dumps({
                "cards": [
                    {
                        "f": "To je <b>nouzový plán</b>.",
                        "t": "Это план Б.",
                        "g": "nouzový - прил.",
                    }
                ]
            })
        }
        mock_post.return_value = mock_resp

        phrases = [{"phrase": "nouzový plán", "context": "To je nouzový plán."}]
        cards = generate_cards(
            phrases,
            use_local_ollama=False,
            fallback_to_ollama=True,
            api_key="fake-key",
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["f"], "To je <b>nouzový plán</b>.")
        self.assertEqual(cards[0]["t"], "Это план Б.")
        mock_client.models.generate_content.assert_called_once()
        mock_post.assert_called_once()


class TestClozeHelpers(unittest.TestCase):
    def test_card_item_parses_hint_field(self):
        item = CardItem(
            f="To je <b>drahá kniha</b>.",
            t="Это дорогая книга.",
            g="drahý - прил.",
            h="дорогая книга",
        )
        self.assertEqual(item.h, "дорогая книга")
        self.assertIn("`h`", _build_prompt([{"phrase": "drahá kniha", "context": "To je drahá kniha."}]))

    def test_format_cloze_sentence_converts_bold(self):
        self.assertEqual(
            format_cloze_sentence("To je <b>drahá kniha</b>."),
            "To je {{c1::drahá kniha}}.",
        )

    def test_format_cloze_sentence_with_hint_and_colon_sanitization(self):
        self.assertEqual(
            format_cloze_sentence("To je <b>drahá kniha</b>.", hint="дорогая: книга"),
            "To je {{c1::drahá kniha::дорогая книга}}.",
        )
        self.assertEqual(
            format_cloze_sentence("To je <b>drahá kniha</b>.", hint="   "),
            "To je {{c1::drahá kniha}}.",
        )

    def test_format_cloze_sentence_wraps_target_phrase(self):
        self.assertEqual(
            format_cloze_sentence("Drahá kniha leží na stole.", "drahá kniha", hint="дорогая книга"),
            "{{c1::Drahá kniha::дорогая книга}} leží na stole.",
        )

    def test_strip_cloze_and_html(self):
        self.assertEqual(
            strip_cloze_and_html("To je {{c1::drahá kniha::дорогая книга}}."),
            "To je drahá kniha.",
        )
        self.assertEqual(
            strip_cloze_and_html("Sraz je v {{c1::14:30::полтретьего}}."),
            "Sraz je v 14:30.",
        )
        self.assertEqual(
            strip_cloze_and_html("To je <b>drahá kniha</b>."),
            "To je drahá kniha.",
        )


if __name__ == "__main__":
    unittest.main()
