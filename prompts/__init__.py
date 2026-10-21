"""현재 Unreal RAG 실행 경로의 프롬프트."""

from pathlib import Path

_DIR = Path(__file__).resolve().parent
SUPERVISOR_PROMPT = (_DIR / "supervisor.md").read_text(encoding="utf-8")
WRITER_PROMPT = (_DIR / "writer.md").read_text(encoding="utf-8")
DIRECT_ANSWER_PROMPT = (_DIR / "direct_answer.md").read_text(encoding="utf-8")
ORCHESTRATOR_PROMPT = (_DIR / "orchestrator.md").read_text(encoding="utf-8")
EVIDENCE_ASSESSOR_PROMPT = (_DIR / "evidence_assessor.md").read_text(encoding="utf-8")
ANSWER_ASSESSOR_PROMPT = (_DIR / "answer_assessor.md").read_text(encoding="utf-8")
