import pytest

from agents.graph import build_graph
from agents.state import Route
from tests.test_graph import EVIDENCE, FakeLLM


def answer_result(text):
    evidence = [EVIDENCE, {**EVIDENCE, "doc_id": "second", "chunk_id": 1, "content": "두 번째 검색 원문"}]
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")], text)
    result = build_graph(model, lambda *args, **kwargs: evidence).invoke({"question": "배열 사용법"})
    assert model.writer_calls == 1
    return result


@pytest.mark.parametrize("text,numbers", [
    pytest.param("`Values[0]`을 사용합니다. [1]", [1], id="inline-index"),
    pytest.param("```cpp\nValues[0] = 1; // [99]\n```\n근거 설명 [1]", [1], id="fenced-code"),
    pytest.param("~~~cpp\nValues[99];\n~~~\n설명 [1]", [1], id="tilde-fence"),
    pytest.param("````cpp\n```\nValues[0];\n```\n````\n설명 [1]", [1], id="longer-fence"),
    pytest.param("설명 [1]\n```cpp\nValues[99];", [1], id="unclosed-fence-after-citation"),
    pytest.param("    Values[0]; // [99]\n\n설명 [1]", [1], id="indented-code"),
    pytest.param("> ```cpp\n> Values[99];\n> ```\n\n설명 [1]", [1], id="quoted-fence"),
    pytest.param("- 코드 예시\n\n  ```cpp\n  Values[99];\n  ```\n\n설명 [1]", [1], id="list-fence"),
    pytest.param("``Values[0] = `value`;`` 설명 [1]", [1], id="multiple-backticks"),
    pytest.param("`Values[99]\n다음 줄` 설명 [1]", [1], id="multiline-inline-code"),
    pytest.param("설명 [2] `Values[1]`", [2], id="code-index-is-not-another-source"),
    pytest.param("설명 **[1]** 및 [2][1]", [1, 2], id="emphasis-and-adjacent-citations"),
    pytest.param("[[99]](https://example.com) 설명 [1]", [1], id="linked-numbers"),
    pytest.param("![[99]](https://example.com/image.png) 설명 [1]", [1], id="image-alt"),
    pytest.param("[자료](https://example.com/[99]) 설명 [1]", [1], id="link-target"),
    pytest.param("설명 [1]\n\n[자료]: https://example.com/[99]", [1], id="reference-link-definition"),
])
def test_code_and_link_numbers_do_not_affect_valid_prose_citations(text, numbers):
    result = answer_result(text)
    assert result["status"] == "done"
    assert result["answer"] == text
    assert [source["number"] for source in result["sources"]] == numbers


@pytest.mark.parametrize("text", [
    pytest.param("`Values[1]`", id="inline-code-only"),
    pytest.param("```cpp\nValues[1];\n```", id="fenced-code-only"),
    pytest.param("    Values[1];", id="indented-code-only"),
    pytest.param("[1](https://example.com)", id="numeric-link-only"),
    pytest.param("[[1]](https://example.com)", id="nested-numeric-label"),
    pytest.param("![[1]](https://example.com/image.png)", id="image-alt-only"),
    pytest.param("```cpp\nValues[0];\n설명 [1]", id="unclosed-fence-no-prose-citation"),
    pytest.param("`Values[0] 설명 [1]", id="unclosed-inline-is-plain-text"),
    pytest.param("`Values[1]` 설명 [99]", id="invalid-prose-citation"),
    pytest.param("설명 [1] 및 [3]", id="mixed-valid-and-invalid"),
    pytest.param("<div>[99]</div> 설명 [1]", id="html-remains-plain-text"),
    pytest.param("설명 [01] 및 [1]", id="noncanonical-leading-zero"),
    pytest.param("설명 [١] 및 [1]", id="arabic-indic-digit"),
    pytest.param("설명 [１] 및 [1]", id="fullwidth-digit"),
    pytest.param("설명 [" + "9" * 5000 + "]", id="oversized-citation-no-int-conversion"),
])
def test_missing_or_noncanonical_prose_citations_are_rejected(text):
    result = answer_result(text)
    assert result["status"] == "invalid_citations"
    assert result["sources"] == []
    assert result["answer"] != text
