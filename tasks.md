# CZECH_HARVESTER: Work Plan

## Current Status: Completed

---

### Step 1: Tests for `doc_parser.py`
- [x] Create `tests/test_doc_parser.py`.
- [x] Cover multi-colored `textRun` concatenation (colored vowels inside words).
- [x] Cover sentence boundary extraction (`context`) across full paragraphs.
- [x] Cover generation of `UpdateTextStyleRequest` highlight clear requests.
- **Verification**: `pytest tests/test_doc_parser.py` or `python -m unittest tests/test_doc_parser.py` exits with code 0.

---

### Step 2: Adaptation of `chunk_tracker.py` for Dynamic Phrases
- [x] Refactor `src/chunk_tracker.py` to decouple from hardcoded `chunk_registry.md`.
- [x] Allow registration and weight updates for arbitrary harvested Czech phrases.
- [x] Ensure persistence into `data/chunk_memory.json`.
- [x] Add unit tests in `tests/test_chunk_tracker.py`.
- **Verification**: Run `python -m unittest tests/test_chunk_tracker.py` with code 0.

---

### Step 3: Card Generation Module (Gemini API + Ollama Router)
- [x] Create `src/card_generator.py` using `google-genai`.
- [x] Implement prompt and structured schema for compact output:
  - `f`: front (target Czech sentence / extracted phrase)
  - `t`: translation
  - `g`: grammar / usage notes
- [x] Add fallback/routing logic for local Ollama endpoint.
- [x] Add unit/mock tests in `tests/test_card_generator.py`.
- **Verification**: Script and tests pass with mock/live calls.

---

### Step 4: Local Audio Transcription & Vowel Highlighting
- [x] Create `src/transcriber.py` powered by `faster-whisper`.
- [x] Implement Czech vowel length detection and formatting logic.
- [x] Prepare Google Docs compatible formatting payload.
- **Verification**: Run transcription on a sample audio clip and verify output text structure.

---

### Step 5: Telegram Worker `/harvest` with Anki Export
- [x] Create `src/harvest_worker.py` with Telegram command handler `/harvest`.
- [x] Pipeline integration: Google Docs parse -> Gemini card generation -> `chunk_memory.json` update -> Anki export.
- [x] Implement export via AnkiConnect HTTP API and fallback `.apkg` generator.
- **Verification**: Trigger `/harvest` flow and verify card creation in Anki and state update in `data/chunk_memory.json`.
