from __future__ import annotations

import datetime
import json
from pathlib import Path
import tempfile
import unittest

from src.chunk_tracker import (
    CHUNK_MEMORY_PATH,
    ChunkMemoryValidationError,
    _calculate_interval,
    _default_chunk_record,
    _sanitize_chunk_record,
    _today,
    compute_chunk_id,
    ensure_chunk_record,
    find_chunk_by_phrase,
    generate_chunk_id,
    get_chunk,
    load_chunk_memory,
    register_harvested_phrases,
    register_phrase,
    register_used_chunks,
    resolve_chunk_id,
    save_chunk_memory,
    update_chunk_on_review,
    update_chunk_on_success,
)


class TestChunkIdGeneration(unittest.TestCase):
    def test_deterministic_chunk_id(self):
        phrase = "mít se dobře"
        id1 = generate_chunk_id(phrase)
        id2 = generate_chunk_id(phrase)
        self.assertEqual(id1, id2)
        self.assertTrue(id1.startswith("chunk_"))
        self.assertEqual(len(id1), len("chunk_") + 12)

    def test_alias_compute_chunk_id(self):
        phrase = "drahá kniha"
        self.assertEqual(generate_chunk_id(phrase), compute_chunk_id(phrase))

    def test_casing_and_whitespace_normalization(self):
        id1 = generate_chunk_id("mít se dobře")
        id2 = generate_chunk_id("  Mít   SE  dobře  ")
        self.assertEqual(id1, id2)

    def test_distinct_phrases_have_distinct_ids(self):
        id1 = generate_chunk_id("mít se dobře")
        id2 = generate_chunk_id("drahá kniha")
        self.assertNotEqual(id1, id2)

    def test_czech_diacritics(self):
        phrase = "příliš žluťoučký kůň úpěl ďábelské ódy"
        cid = generate_chunk_id(phrase)
        self.assertTrue(cid.startswith("chunk_"))
        self.assertEqual(cid, generate_chunk_id(phrase))

    def test_empty_or_whitespace_raises_validation_error(self):
        with self.assertRaises(ChunkMemoryValidationError):
            generate_chunk_id("")
        with self.assertRaises(ChunkMemoryValidationError):
            generate_chunk_id("   ")


class TestRecordDefaultsAndSanitization(unittest.TestCase):
    def test_default_record_creation(self):
        record = _default_chunk_record("mít se dobře", initial_weight=0.2)
        self.assertEqual(record["phrase"], "mít se dobře")
        self.assertEqual(record["weight"], 0.2)
        self.assertEqual(record["level"], 0.2)
        self.assertEqual(record["usage_count"], 0)
        self.assertEqual(record["success_rate"], 0.0)
        self.assertEqual(record["interval_days"], 1)
        self.assertEqual(record["first_seen"], _today())
        self.assertEqual(record["last_seen"], _today())
        self.assertIn("mít se dobře", record["variation_forms"])

    def test_sanitize_missing_fields_and_clamping(self):
        raw = {
            "weight": 2.5,
            "success_rate": -1.0,
            "usage_count": -5,
            "phrase": "  dobrý den  ",
            "variation_forms": ["dobrý den", "  dobrý večer  ", ""],
        }
        sanitized = _sanitize_chunk_record(raw)
        self.assertEqual(sanitized["weight"], 1.0)
        self.assertEqual(sanitized["success_rate"], 0.0)
        self.assertEqual(sanitized["usage_count"], 0)
        self.assertEqual(sanitized["phrase"], "dobrý den")
        self.assertEqual(sanitized["variation_forms"], ["dobrý den", "dobrý večer"])

    def test_interval_calculation(self):
        self.assertEqual(_calculate_interval(0.1, 0), 1)
        self.assertEqual(_calculate_interval(0.1, 1), 1)
        self.assertEqual(_calculate_interval(0.3, 3), 3)
        self.assertEqual(_calculate_interval(0.5, 5), 7)
        self.assertEqual(_calculate_interval(0.7, 8), 14)
        self.assertEqual(_calculate_interval(0.9, 15), 30)


class TestEnsureAndResolveChunk(unittest.TestCase):
    def setUp(self):
        self.memory = {}

    def test_ensure_chunk_record_with_new_phrase(self):
        phrase = "udělat si čas"
        chunk_id = ensure_chunk_record(self.memory, phrase, initial_weight=0.1)
        self.assertIn(chunk_id, self.memory)
        record = self.memory[chunk_id]
        self.assertEqual(record["phrase"], phrase)
        self.assertEqual(record["weight"], 0.1)
        self.assertEqual(record["usage_count"], 0)

    def test_ensure_chunk_record_idempotent(self):
        phrase = "udělat si čas"
        cid1 = ensure_chunk_record(self.memory, phrase)
        self.memory[cid1]["usage_count"] = 5
        self.memory[cid1]["weight"] = 0.5

        # Re-ensuring must not reset usage_count or weight
        cid2 = ensure_chunk_record(self.memory, phrase)
        self.assertEqual(cid1, cid2)
        self.assertEqual(self.memory[cid1]["usage_count"], 5)
        self.assertEqual(self.memory[cid1]["weight"], 0.5)

    def test_ensure_with_explicit_chunk_id(self):
        cid = "chunk_custom_001"
        res_id = ensure_chunk_record(self.memory, cid, phrase="vlastní fráze")
        self.assertEqual(res_id, cid)
        self.assertEqual(self.memory[cid]["phrase"], "vlastní fráze")

    def test_resolve_chunk_id_with_existing_phrase(self):
        phrase = "mít pravdu"
        cid = ensure_chunk_record(self.memory, phrase)
        # Calling resolve on uppercase version should resolve to the same chunk_id
        resolved = resolve_chunk_id(self.memory, "MÍT PRAVDU")
        self.assertEqual(resolved, cid)

    def test_ensure_empty_raises_error(self):
        with self.assertRaises(ChunkMemoryValidationError):
            ensure_chunk_record(self.memory, "")


class TestUpdateChunkStatistics(unittest.TestCase):
    def setUp(self):
        self.memory = {}

    def test_update_chunk_on_success_new_phrase(self):
        phrase = "dostat nápad"
        chunk_id = update_chunk_on_success(self.memory, phrase)
        record = self.memory[chunk_id]

        self.assertEqual(record["usage_count"], 1)
        self.assertEqual(record["weight"], 0.1)
        self.assertEqual(record["success_rate"], 1.0)
        self.assertEqual(record["level"], 0.1)

    def test_repeated_success_updates(self):
        phrase = "dostat nápad"
        chunk_id = update_chunk_on_success(self.memory, phrase)
        update_chunk_on_success(self.memory, phrase)
        record = self.memory[chunk_id]

        self.assertEqual(record["usage_count"], 2)
        self.assertEqual(record["weight"], 0.2)
        self.assertEqual(record["success_rate"], 1.0)
        self.assertEqual(record["level"], 0.2)

    def test_weight_capped_at_one(self):
        phrase = "časté slovo"
        chunk_id = ensure_chunk_record(self.memory, phrase, initial_weight=0.95)
        update_chunk_on_success(self.memory, phrase)
        self.assertEqual(self.memory[chunk_id]["weight"], 1.0)

    def test_update_chunk_on_review_failure(self):
        phrase = "těžká vazba"
        chunk_id = update_chunk_on_success(self.memory, phrase)  # count 1, weight 0.1, rate 1.0
        update_chunk_on_review(self.memory, phrase, success=False)  # count 2, weight 0.0, rate 0.5

        record = self.memory[chunk_id]
        self.assertEqual(record["usage_count"], 2)
        self.assertEqual(record["weight"], 0.0)
        self.assertEqual(record["success_rate"], 0.5)


class TestLookupHelpers(unittest.TestCase):
    def setUp(self):
        self.memory = {}
        self.cid = ensure_chunk_record(self.memory, "dávat pozor")

    def test_find_chunk_by_phrase(self):
        res = find_chunk_by_phrase(self.memory, "dávat pozor")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], self.cid)

        # Case-insensitive
        res_case = find_chunk_by_phrase(self.memory, "DÁVAT POZOR")
        self.assertIsNotNone(res_case)
        self.assertEqual(res_case[0], self.cid)

        # Nonexistent phrase
        self.assertIsNone(find_chunk_by_phrase(self.memory, "neexistující fráze"))

    def test_get_chunk(self):
        # Lookup by chunk_id
        record1 = get_chunk(self.memory, self.cid)
        self.assertIsNotNone(record1)
        self.assertEqual(record1["phrase"], "dávat pozor")

        # Lookup by phrase string
        record2 = get_chunk(self.memory, "dávat pozor")
        self.assertIsNotNone(record2)
        self.assertEqual(record2["phrase"], "dávat pozor")

        # Nonexistent
        self.assertIsNone(get_chunk(self.memory, "neznámé"))


class TestPersistenceAndDynamicPhrases(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "test_chunk_memory.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_roundtrip(self):
        memory = {}
        cid1 = update_chunk_on_success(memory, "mít hlad")
        cid2 = update_chunk_on_success(memory, "mít žízeň")

        save_chunk_memory(memory, path=self.temp_file)
        loaded = load_chunk_memory(path=self.temp_file)

        self.assertEqual(len(loaded), 2)
        self.assertIn(cid1, loaded)
        self.assertIn(cid2, loaded)
        self.assertEqual(loaded[cid1]["phrase"], "mít hlad")
        self.assertEqual(loaded[cid2]["phrase"], "mít žízeň")

    def test_load_nonexistent_file_returns_empty(self):
        loaded = load_chunk_memory(path=Path(self.temp_dir.name) / "nonexistent.json")
        self.assertEqual(loaded, {})

    def test_load_corrupted_file_raises_error(self):
        corrupt_file = Path(self.temp_dir.name) / "corrupt.json"
        corrupt_file.write_text("{bad json...", encoding="utf-8")
        with self.assertRaises(ChunkMemoryValidationError):
            load_chunk_memory(path=corrupt_file)

    def test_register_phrase_standalone(self):
        cid, record = register_phrase(
            "přijít vhod",
            source_lesson="audiobook_ch1",
            initial_weight=0.1,
            memory_path=self.temp_file,
        )
        self.assertTrue(cid.startswith("chunk_"))
        self.assertEqual(record["phrase"], "přijít vhod")
        self.assertEqual(record["source_lesson"], "audiobook_ch1")

        # Verify disk persistence
        loaded = load_chunk_memory(path=self.temp_file)
        self.assertIn(cid, loaded)
        self.assertEqual(loaded[cid]["phrase"], "přijít vhod")

    def test_register_used_chunks_batch(self):
        phrases = ["dát si pozor", "dělat si legraci", "být v pohodě"]
        updated = register_used_chunks(phrases, memory_path=self.temp_file, source_lesson="audiobook_ch2")

        self.assertEqual(len(updated), 3)
        for p in phrases:
            cid = generate_chunk_id(p)
            self.assertIn(cid, updated)
            self.assertEqual(updated[cid]["phrase"], p)
            self.assertEqual(updated[cid]["usage_count"], 1)
            self.assertEqual(updated[cid]["weight"], 0.1)
            self.assertEqual(updated[cid]["source_lesson"], "audiobook_ch2")

        # Second batch with an overlapping phrase
        register_used_chunks(["být v pohodě", "nová fráze"], memory_path=self.temp_file)
        reloaded = load_chunk_memory(path=self.temp_file)

        self.assertEqual(len(reloaded), 4)
        cid_overlap = generate_chunk_id("být v pohodě")
        self.assertEqual(reloaded[cid_overlap]["usage_count"], 2)
        self.assertEqual(reloaded[cid_overlap]["weight"], 0.2)

    def test_register_harvested_phrases(self):
        harvested = ["představit si", "zvyknout si"]
        updated = register_harvested_phrases(harvested, source="gdoc_export_01", memory_path=self.temp_file)
        self.assertEqual(len(updated), 2)
        for h in harvested:
            found = find_chunk_by_phrase(updated, h)
            self.assertIsNotNone(found)
            self.assertEqual(found[1]["source_lesson"], "gdoc_export_01")


class TestProductionDataCompatibility(unittest.TestCase):
    def test_existing_production_chunk_memory_loads_successfully(self):
        """
        Ensures that data/chunk_memory.json is read properly and all existing chunk IDs
        are retained without any static registry filtering errors.
        """
        if not CHUNK_MEMORY_PATH.exists():
            self.skipTest(f"{CHUNK_MEMORY_PATH} does not exist")

        memory = load_chunk_memory()
        self.assertGreater(len(memory), 0)
        # Verify known sample keys from data/chunk_memory.json
        expected_keys = {"chunk_001", "chunk_002", "chunk_003", "chunk_004", "chunk_005", "chunk_010"}
        for k in expected_keys:
            self.assertIn(k, memory, f"Expected key {k} to be preserved in chunk_memory")
            record = memory[k]
            self.assertIn("usage_count", record)
            self.assertIn("weight", record)
            self.assertIn("schema_version", record)


if __name__ == "__main__":
    unittest.main()
