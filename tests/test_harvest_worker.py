from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from aiogram.types import Message
import requests

from src.chunk_tracker import load_chunk_memory
from src.harvest_worker import (
    DEFAULT_DECK_NAME,
    build_dispatcher,
    create_bot,
    export_to_anki,
    export_via_genanki,
    extract_doc_id,
    fetch_document_ast,
    get_stub_document_ast,
    handle_harvest_command,
    run_harvest_pipeline,
)


class TestDocIdExtraction(unittest.TestCase):
    def test_extract_plain_doc_id(self):
        self.assertEqual(extract_doc_id("my_document_id_123"), "my_document_id_123")
        self.assertEqual(extract_doc_id("  doc-abc_xyz  "), "doc-abc_xyz")

    def test_extract_from_google_docs_url(self):
        url = "https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?tab=t.0#heading=h.abc"
        self.assertEqual(extract_doc_id(url), "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")

    def test_empty_input(self):
        self.assertEqual(extract_doc_id(""), "")
        self.assertEqual(extract_doc_id("   "), "")


class TestFetchDocumentAst(unittest.TestCase):
    def test_stub_ast_structure(self):
        doc_id = "test_doc_001"
        ast = get_stub_document_ast(doc_id)
        self.assertEqual(ast["documentId"], doc_id)
        self.assertIn("body", ast)
        self.assertIn("content", ast["body"])

    def test_fetch_document_ast_without_token_returns_stub(self):
        ast = fetch_document_ast("sample_id", access_token=None)
        self.assertEqual(ast["documentId"], "sample_id")

    def test_fetch_document_ast_with_token(self):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"documentId": "live_doc", "body": {"content": []}}

        with mock.patch("src.harvest_worker.requests.get", return_value=mock_resp) as mock_get:
            ast = fetch_document_ast("live_doc", access_token="mock_token_123")
            self.assertEqual(ast["documentId"], "live_doc")
            mock_get.assert_called_once_with(
                "https://docs.googleapis.com/v1/documents/live_doc",
                headers={"Authorization": "Bearer mock_token_123"},
                timeout=10.0,
            )


class TestExportToAnki(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.sample_cards = [
            {
                "f": "To je <b>drahá kniha</b>.",
                "t": "Это дорогая книга.",
                "g": "drahý - прил., kniha - сущ.",
            },
            {
                "f": "Mám se <b>dobře</b>.",
                "t": "У меня все хорошо.",
                "g": "dobře - наречие",
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _mock_rpc_resp(result):
        resp = mock.MagicMock(status_code=200)
        resp.json.return_value = {"result": result, "error": None}
        return resp

    def test_default_deck_name(self):
        self.assertEqual(DEFAULT_DECK_NAME, "Čeština::Harvester")

    def test_empty_cards_export(self):
        res = export_to_anki([], output_dir=self.output_dir)
        self.assertEqual(res["method"], "none")
        self.assertEqual(res["count"], 0)
        self.assertIsNone(res["apkg_path"])

    def test_ankiconnect_cloze_english_fields_with_audio(self):
        responses = [
            self._mock_rpc_resp(12345),  # createDeck
            self._mock_rpc_resp(["Text", "Back Extra"]),  # modelFieldNames
            self._mock_rpc_resp([]),  # findNotes card 1
            self._mock_rpc_resp("czech_1.mp3"),  # storeMediaFile card 1
            self._mock_rpc_resp([]),  # findNotes card 2
            self._mock_rpc_resp("czech_2.mp3"),  # storeMediaFile card 2
            self._mock_rpc_resp([101, 102]),  # addNotes
        ]

        with mock.patch("src.harvest_worker.synthesize_czech_audio", return_value=b"MP3BYTES") as mock_tts, \
             mock.patch("src.harvest_worker.requests.post", side_effect=responses) as mock_post:
            res = export_to_anki(self.sample_cards, output_dir=self.output_dir)

            self.assertEqual(res["method"], "ankiconnect")
            self.assertEqual(res["count"], 2)
            self.assertIsNone(res["apkg_path"])
            self.assertEqual(mock_tts.call_count, 2)
            # Clean sentence without {{c1::}} passed to TTS
            self.assertEqual(mock_tts.call_args_list[0].args[0], "To je drahá kniha.")

            # Verify addNotes payload
            add_notes_payload = mock_post.call_args_list[-1].kwargs["json"]
            self.assertEqual(add_notes_payload["action"], "addNotes")
            note0 = add_notes_payload["params"]["notes"][0]
            self.assertEqual(note0["deckName"], "Čeština::Harvester")
            self.assertEqual(note0["modelName"], "Cloze")
            self.assertEqual(note0["fields"]["Text"], "To je {{c1::drahá kniha}}.")
            self.assertIn("Это дорогая книга.", note0["fields"]["Back Extra"])
            self.assertIn("[sound:czech_", note0["fields"]["Back Extra"])
            self.assertNotIn("<br>", note0["fields"]["Back Extra"])

            # Verify findNotes used clean sentence without {{c1::}}
            find_payload = mock_post.call_args_list[2].kwargs["json"]
            self.assertEqual(find_payload["action"], "findNotes")
            self.assertIn("To je drahá kniha.", find_payload["params"]["query"])
            self.assertNotIn("{{c1::", find_payload["params"]["query"])

    def test_ankiconnect_cloze_russian_localized_fields(self):
        responses = [
            self._mock_rpc_resp(12345),  # createDeck
            self._mock_rpc_resp(["Текст", "Дополнительно"]),  # Russian Cloze fields
            self._mock_rpc_resp([]),  # findNotes
            self._mock_rpc_resp("czech_1.mp3"),  # storeMediaFile
            self._mock_rpc_resp([101]),  # addNotes
        ]

        with mock.patch("src.harvest_worker.synthesize_czech_audio", return_value=b"MP3"), \
             mock.patch("src.harvest_worker.requests.post", side_effect=responses) as mock_post:
            res = export_to_anki([self.sample_cards[0]], output_dir=self.output_dir)

            self.assertEqual(res["method"], "ankiconnect")
            self.assertEqual(res["count"], 1)
            note0 = mock_post.call_args_list[-1].kwargs["json"]["params"]["notes"][0]
            self.assertIn("Текст", note0["fields"])
            self.assertIn("Дополнительно", note0["fields"])
            self.assertEqual(note0["fields"]["Текст"], "To je {{c1::drahá kniha}}.")

    def test_ankiconnect_skips_duplicate_via_find_notes(self):
        responses = [
            self._mock_rpc_resp(12345),  # createDeck
            self._mock_rpc_resp(["Text", "Back Extra"]),  # modelFieldNames
            self._mock_rpc_resp([999]),  # findNotes -> duplicate for card 1!
            self._mock_rpc_resp([]),  # findNotes -> new for card 2
            self._mock_rpc_resp("czech_2.mp3"),  # storeMediaFile card 2
            self._mock_rpc_resp([202]),  # addNotes
        ]

        with mock.patch("src.harvest_worker.synthesize_czech_audio", return_value=b"MP3"), \
             mock.patch("src.harvest_worker.requests.post", side_effect=responses) as mock_post:
            res = export_to_anki(self.sample_cards, output_dir=self.output_dir)

            self.assertEqual(res["method"], "ankiconnect")
            self.assertEqual(res["count"], 1)
            self.assertEqual(res["skipped_duplicates"], 1)
            notes = mock_post.call_args_list[-1].kwargs["json"]["params"]["notes"]
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0]["fields"]["Text"], "Mám se {{c1::dobře}}.")

    def test_ankiconnect_all_duplicates_skips_add_notes_and_genanki(self):
        responses = [
            self._mock_rpc_resp(12345),  # createDeck
            self._mock_rpc_resp(["Text", "Back Extra"]),  # modelFieldNames
            self._mock_rpc_resp([999]),  # findNotes card 1 -> duplicate
            self._mock_rpc_resp([888]),  # findNotes card 2 -> duplicate
        ]

        with mock.patch("src.harvest_worker.requests.post", side_effect=responses):
            res = export_to_anki(self.sample_cards, output_dir=self.output_dir)

            self.assertEqual(res["method"], "ankiconnect")
            self.assertEqual(res["count"], 0)
            self.assertEqual(res["skipped_duplicates"], 2)
            self.assertIsNone(res["apkg_path"])

    def test_ankiconnect_connection_error_falls_back_to_genanki(self):
        with mock.patch("src.harvest_worker.requests.post", side_effect=requests.ConnectionError("AnkiConnect unreachable")):
            res = export_to_anki(self.sample_cards, output_dir=self.output_dir)

            self.assertEqual(res["method"], "genanki")
            self.assertEqual(res["count"], 2)
            self.assertIsNotNone(res["apkg_path"])

            apkg_file: Path = res["apkg_path"]
            self.assertTrue(apkg_file.exists())
            self.assertTrue(apkg_file.name.startswith("harvest_"))
            self.assertTrue(apkg_file.name.endswith(".apkg"))
            self.assertGreater(apkg_file.stat().st_size, 0)

    def test_ankiconnect_returns_error_falls_back_to_genanki(self):
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": None, "error": "deck not found"}

        with mock.patch("src.harvest_worker.requests.post", return_value=mock_resp):
            res = export_to_anki(self.sample_cards, output_dir=self.output_dir)

            self.assertEqual(res["method"], "genanki")
            self.assertEqual(res["count"], 2)
            self.assertTrue(res["apkg_path"].exists())

    def test_ankiconnect_all_null_results_falls_back_to_genanki(self):
        responses = [
            self._mock_rpc_resp(123),  # createDeck
            self._mock_rpc_resp(["Text", "Back Extra"]),  # modelFieldNames
            self._mock_rpc_resp([]),  # findNotes 1
            self._mock_rpc_resp("a.mp3"),  # storeMediaFile 1
            self._mock_rpc_resp([]),  # findNotes 2
            self._mock_rpc_resp("b.mp3"),  # storeMediaFile 2
            self._mock_rpc_resp([None, None]),  # addNotes -> all null
        ]

        with mock.patch("src.harvest_worker.synthesize_czech_audio", return_value=b"MP3"), \
             mock.patch("src.harvest_worker.requests.post", side_effect=responses):
            res = export_to_anki(self.sample_cards, output_dir=self.output_dir)

            self.assertEqual(res["method"], "genanki")
            self.assertEqual(res["count"], 2)
            self.assertTrue(res["apkg_path"].exists())

    def test_ankiconnect_partial_null_counts_only_added_notes(self):
        responses = [
            self._mock_rpc_resp(123),  # createDeck
            self._mock_rpc_resp(["Text", "Back Extra"]),  # modelFieldNames
            self._mock_rpc_resp([]),  # findNotes 1
            self._mock_rpc_resp("a.mp3"),  # storeMediaFile 1
            self._mock_rpc_resp([]),  # findNotes 2
            self._mock_rpc_resp("b.mp3"),  # storeMediaFile 2
            self._mock_rpc_resp([101, None]),  # addNotes -> partial null
        ]

        with mock.patch("src.harvest_worker.synthesize_czech_audio", return_value=b"MP3"), \
             mock.patch("src.harvest_worker.requests.post", side_effect=responses):
            res = export_to_anki(self.sample_cards, output_dir=self.output_dir)

            self.assertEqual(res["method"], "ankiconnect")
            self.assertEqual(res["count"], 1)

    def test_export_via_genanki_direct(self):
        apkg_path = export_via_genanki(
            self.sample_cards,
            deck_name=DEFAULT_DECK_NAME,
            output_dir=self.output_dir,
        )
        self.assertTrue(apkg_path.exists())
        self.assertEqual(apkg_path.suffix, ".apkg")


class TestHarvestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.memory_path = self.output_dir / "test_chunk_memory.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_success(self):
        mock_cards = [
            {"f": "Dnes máme <b>krásný den</b> na procházku.", "t": "Сегодня прекрасный день.", "g": "příd. jm."}
        ]
        mock_export_res = {
            "method": "ankiconnect",
            "count": 1,
            "apkg_path": None,
        }

        with mock.patch("src.harvest_worker.generate_cards", return_value=mock_cards) as mock_gen_cards, \
             mock.patch("src.harvest_worker.export_to_anki", return_value=mock_export_res) as mock_export:

            pipeline_res = run_harvest_pipeline(
                doc_id="doc_test_100",
                output_dir=self.output_dir,
                memory_path=self.memory_path,
            )

            self.assertEqual(pipeline_res["status"], "success")
            self.assertEqual(pipeline_res["phrases_count"], 1)
            self.assertEqual(pipeline_res["cards_count"], 1)
            self.assertGreater(len(pipeline_res["clear_requests"]), 0)

            # Verify chunk_memory update
            loaded_memory = load_chunk_memory(path=self.memory_path)
            self.assertEqual(len(loaded_memory), 1)
            first_chunk = list(loaded_memory.values())[0]
            self.assertEqual(first_chunk["phrase"], "krásný den")
            self.assertEqual(first_chunk["source_lesson"], "doc_test_100")

    def test_pipeline_skips_duplicates_in_chunk_memory(self):
        from src.chunk_tracker import register_harvested_phrases

        register_harvested_phrases(["krásný den"], source="old_doc", memory_path=self.memory_path)

        with mock.patch("src.harvest_worker.generate_cards") as mock_gen_cards, \
             mock.patch("src.harvest_worker.export_to_anki", return_value={"method": "none", "count": 0, "apkg_path": None}):
            pipeline_res = run_harvest_pipeline(
                doc_id="doc_dup_test",
                output_dir=self.output_dir,
                memory_path=self.memory_path,
            )

            self.assertEqual(pipeline_res["status"], "success")
            self.assertEqual(pipeline_res["phrases_count"], 1)
            self.assertEqual(pipeline_res["cards_count"], 0)
            mock_gen_cards.assert_not_called()

    def test_pipeline_with_empty_document(self):
        empty_ast = {"documentId": "empty_doc", "body": {"content": []}}
        pipeline_res = run_harvest_pipeline(
            doc_id="empty_doc",
            custom_doc_ast=empty_ast,
            output_dir=self.output_dir,
            memory_path=self.memory_path,
        )
        self.assertEqual(pipeline_res["status"], "empty")
        self.assertEqual(pipeline_res["phrases_count"], 0)


def _run(coro):
    """Executes a simple coroutine chain without requiring loopback socket binding."""
    try:
        while True:
            coro.send(None)
    except StopIteration as e:
        return e.value


class DummyStatusMessage:
    def __init__(self):
        self.edits: list[str] = []

    async def edit_text(self, text: str, **kwargs):
        self.edits.append(text)


class DummyMessage:
    def __init__(self, text: str):
        self.text = text
        self.answers: list[str] = []
        self.documents: list[Any] = []
        self.status_msg = DummyStatusMessage()

    async def answer(self, text: str, **kwargs):
        self.answers.append(text)
        return self.status_msg

    async def answer_document(self, document: Any, **kwargs):
        self.documents.append({"document": document, "kwargs": kwargs})


class TestTelegramHandler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_handle_harvest_missing_argument(self):
        msg = DummyMessage("/harvest")
        _run(handle_harvest_command(msg))
        self.assertEqual(len(msg.answers), 1)
        self.assertIn("укажите ID документа", msg.answers[0])

    def test_handle_harvest_ankiconnect_success(self):
        msg = DummyMessage("/harvest doc_abc_123")

        pipeline_mock_res = {
            "status": "success",
            "doc_id": "doc_abc_123",
            "phrases_count": 3,
            "cards_count": 3,
            "export_result": {"method": "ankiconnect", "count": 3, "apkg_path": None},
        }

        with mock.patch("src.harvest_worker.run_harvest_pipeline", return_value=pipeline_mock_res):
            _run(handle_harvest_command(msg))

            self.assertEqual(len(msg.answers), 1)
            self.assertEqual(len(msg.status_msg.edits), 1)
            self.assertIn("Успешно обработано", msg.status_msg.edits[0])
            self.assertIn("AnkiConnect", msg.status_msg.edits[0])

    def test_handle_harvest_genanki_fallback_sends_document(self):
        msg = DummyMessage("/harvest doc_xyz_789")

        fake_apkg = self.output_dir / "harvest_2026-11-07.apkg"
        fake_apkg.write_text("fake binary", encoding="utf-8")

        pipeline_mock_res = {
            "status": "success",
            "doc_id": "doc_xyz_789",
            "phrases_count": 2,
            "cards_count": 2,
            "export_result": {"method": "genanki", "count": 2, "apkg_path": fake_apkg},
        }

        with mock.patch("src.harvest_worker.run_harvest_pipeline", return_value=pipeline_mock_res):
            _run(handle_harvest_command(msg))

            self.assertEqual(len(msg.status_msg.edits), 1)
            self.assertEqual(len(msg.documents), 1)
            caption = msg.documents[0]["kwargs"].get("caption", "")
            self.assertIn(".apkg", caption)

    def test_handle_harvest_empty_document(self):
        msg = DummyMessage("/harvest empty_doc")

        pipeline_mock_res = {
            "status": "empty",
            "doc_id": "empty_doc",
            "phrases_count": 0,
            "cards_count": 0,
            "export_result": {"method": "none", "count": 0, "apkg_path": None},
        }

        with mock.patch("src.harvest_worker.run_harvest_pipeline", return_value=pipeline_mock_res):
            _run(handle_harvest_command(msg))

            self.assertEqual(len(msg.status_msg.edits), 1)
            self.assertIn("не найдено желтых выделений", msg.status_msg.edits[0])


class TestBotAndDispatcherCreation(unittest.TestCase):
    @mock.patch("src.harvest_worker.Bot")
    def test_create_bot_and_dispatcher(self, mock_bot_cls):
        mock_bot = mock.MagicMock()
        mock_bot_cls.return_value = mock_bot

        bot = create_bot("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
        self.assertEqual(bot, mock_bot)
        mock_bot_cls.assert_called_once_with(token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")

        dp = build_dispatcher()
        self.assertIsNotNone(dp)


if __name__ == "__main__":
    unittest.main()
