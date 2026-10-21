"""Unreal Engine RAG CLI. 웹 UI: uvicorn api:app --host 127.0.0.1 --port 8000."""

import argparse
import os
import sys
from getpass import getpass

from dotenv import load_dotenv
from langchain_core.tracers.langchain import wait_for_all_tracers

load_dotenv()

from agents import build_graph  # noqa: E402
from agents.docs import CorpusChangedError  # noqa: E402
from agents.graph import run_config  # noqa: E402
from agents.llm import create_llm  # noqa: E402


def _engine_version(value: str) -> str:
    version = value.strip()
    if not version or len(version) > 50:
        raise argparse.ArgumentTypeError("엔진 버전은 공백 제거 후 1~50자여야 합니다.")
    return version


def main():
    parser = argparse.ArgumentParser(description="Unreal Engine RAG CLI")
    parser.add_argument("--engine-version", type=_engine_version, help="같은 마이너 버전 계열 검색 (5.4는 5.4.x 포함)")
    parser.add_argument("question", nargs="*", help="질문 (생략하면 대화 모드)")
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or getpass("LLM API 키: ").strip()
    if not api_key:
        print("API 키가 필요합니다.")
        return 1
    graph = build_graph(create_llm(api_key))
    history = []

    def ask(question):
        result = graph.invoke({"question": question, "history": history[-10:],
                               "engine_version": args.engine_version},
                              config=run_config("cli", engine_version=args.engine_version))
        print("\n" + result["answer"])
        if result["status"] == "invalid_citations":
            return 1
        for source in result["sources"]:
            print(f"[{source['number']}] {source['title']} | UE {source.get('engine_version', '미지정')}")
        history.extend([{"role": "user", "content": question},
                        {"role": "assistant", "content": result["answer"][:8000]}])
        return 0

    try:
        if args.question:
            return ask(" ".join(args.question))
        print(f"Tsukuyomi for Unreal — 검색 버전: {args.engine_version or '전체'} — 종료: exit")
        while True:
            question = input("\n질문> ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                return 0
            if question:
                ask(question)
    except (EOFError, KeyboardInterrupt):
        return 0
    except CorpusChangedError:
        print("답변을 만들던 중 지식이 변경되었습니다. 같은 질문을 다시 보내 주세요.")
        return 1
    except Exception:
        print("요청에 실패했습니다. API 키, 모델 설정, 문서 인덱싱 환경을 확인해 주세요.")
        return 1
    finally:
        wait_for_all_tracers()  # 단일 질문 실행이 LangSmith 전송 전에 끝나지 않게 한다.

if __name__ == "__main__":
    sys.exit(main())
