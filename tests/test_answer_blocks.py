import pytest

from api import _answer_blocks


@pytest.mark.parametrize("text", ["", "일반 답변 [1]", "인라인 `Values[0]` [1]", "<pre>literal</pre> [1]"])
def test_no_block_code_needs_no_extra_display_data(text):
    assert _answer_blocks(text) == []


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_fenced_code_preserves_surrounding_text_and_canonical_code_newlines(newline):
    text = newline.join(["설명 [1]", "", "```cpp options", "    Values[0] = 1;", "```", "", "끝"])

    assert _answer_blocks(text) == [
        {"type": "text", "content": "설명 [1]" + newline * 2},
        {"type": "code", "content": "    Values[0] = 1;\n", "language": "cpp"},
        {"type": "text", "content": newline + "끝"},
    ]


def test_multiple_fences_keep_order_and_plain_text():
    assert _answer_blocks("```cpp\nF();\n```\n설명 [1]\n~~~\nG();\n~~~") == [
        {"type": "code", "content": "F();\n", "language": "cpp"},
        {"type": "text", "content": "설명 [1]\n"},
        {"type": "code", "content": "G();\n", "language": ""},
    ]


@pytest.mark.parametrize("text, code", [
    ("설명 [1]\n\n    F();\n", "F();\n"),
    ("> 설명 [1]\n>\n> ```cpp\n> F();\n> ```\n", "F();\n"),
    ("1. 설명 [1]\n\n    ```cpp\n    F();\n    ```\n", "F();\n"),
    ("설명 [1]\n~~~cpp\nF();", "F();"),
])
def test_indented_nested_and_unclosed_code_is_displayed_as_code(text, code):
    blocks = _answer_blocks(text)
    assert [block["content"] for block in blocks if block["type"] == "code"] == [code]
    assert "설명 [1]" in blocks[0]["content"]


def test_html_and_unicode_separators_are_plain_data():
    before = "전제\u2028확인 [1]\n"
    code = '<script>alert(1)</script>\n<img src=x onerror="alert(1)">\n'
    blocks = _answer_blocks(before + "```html\n" + code + "```\n끝")
    assert blocks[0] == {"type": "text", "content": before}
    assert blocks[1] == {"type": "code", "content": code, "language": "html"}
    assert blocks[2] == {"type": "text", "content": "끝"}
