# Repository Guidelines

## Work Planning
Before each task, write a plan covering scope, affected files, validation, and open questions. Review it against these guidelines and the latest user requirements before implementation. Update and recheck it when scope changes. Record unknowns and decisions requiring input as TODOs. Keep changes minimal and documentation concise.

## Project Structure & Direction
The current application evaluates startup investments. The target is an internal Unreal Engine RAG chatbot with a Vue.js frontend and a Python/LangGraph backend using the Supervisor pattern.

- `app.py`: current CLI entry point.
- `agents/graph.py`, `state.py`, `tools.py`: orchestration, state, and tools.
- `agents/docs.py`: ingestion, FAISS indexing, and retrieval.
- `prompts/`: agent instructions; update alongside changed roles.
- `data/docs/`: source PDFs and `metadata.json`.
- `data/vectorstores/BAAI--bge-m3/`: generated retrieval index.
- `outputs/`: reports and execution traces.
- `frontend/`: proposed Vue location; not yet implemented.

Reuse retrieval and Supervisor infrastructure. Replace investment-specific behavior only as required. Keep `BAAI/bge-m3` for local embeddings. Store Unreal Engine techniques and development errors with their solutions; retain engine version, context, reproduction steps, and references. Answers should cite retrieved evidence and identify missing information.

## Development Commands
Run from the repository root with the project's Python environment active:

- `python -m pip install -r requirements.txt`: install dependencies.
- `python -m agents.docs`: rebuild the index after document changes; replaces generated index files.
- `python app.py`: run the existing interactive CLI.
- `python app.py "question"`: execute one question using current behavior.

Document Vue commands when its package configuration exists.

## Coding Style & Testing
Use four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and existing type annotations. No formatter, linter, test framework, or coverage threshold is configured.

Validate changed retrieval, source attribution, Supervisor routing, retry limits, and insufficient-evidence behavior. Add focused regression tests for behavioral changes; use `tests/test_*.py` for Python tests. Document verification commands and results.

## Configuration & Reviews
Use server-side API keys for external LLM calls; never commit keys or expose them through Vue bundles or traces. Current configuration uses `.env` with `OPENAI_API_KEY` and optional `OPENAI_MODEL`.

Git history is unavailable in this checkout. Use concise imperative commits. PRs should explain scope, validation, unresolved questions, linked issues where applicable, and screenshots for UI changes.

## Open Decisions
TODO: Confirm LLM providers, backend API framework, internal authentication, supported document formats, and knowledge submission/review workflow before implementing dependent features.
