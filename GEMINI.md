# CZECH_HARVESTER: Sentence Mining Pipeline

## Architecture: Audio -> Google Docs -> Gemini -> Anki
Automated pipeline for conversational Czech sentence mining and spaced repetition.

1. **Local STT**: `faster-whisper` transcribes audio recordings into structured text with vowel length markings.
2. **Google Docs Sync & Parsing**: `src/doc_parser.py` extracts yellow-highlighted phrases, extracts sentence context, and clears highlight markers via Google Docs API `UpdateTextStyleRequest`.
3. **Card Generation**: Batch processing via `google-genai` (with local Ollama fallback/router) outputting compact JSON cards:
   - `f`: front (target Czech sentence / extracted phrase)
   - `t`: translation (target meaning)
   - `g`: grammar note (morphology, aspect, case, context)
4. **Learning State**: `data/chunk_memory.json` tracks phrase frequency, weights, and intervals.
5. **Anki Integration**: Automated deck export via AnkiConnect (HTTP API) and standalone `.apkg` files.

## Project Structure
- `src/` — core modules (`doc_parser.py`, `chunk_tracker.py`, generators, workers).
- `data/` — runtime state (`chunk_memory.json`), harvested dumps.
- `tests/` — unit & integration test suites.
- `.agents/` — MCP and agent configurations (`mcp_config.json`).

## Operational Rules & Testing Discipline
- **Autonomous Testing**: The agent must independently execute tests (`pytest` / `python -m unittest`) via terminal commands and fix all failures until exit code 0 is reached.
- **Zero Manual Copy-Paste**: Never ask the user to manually copy error traces, run tests, or apply routine code edits when the agent can run and edit them directly.
- **Codebase Integrity**: Keep modules focused, type-annotated, and covered by automated tests under `tests/`.
