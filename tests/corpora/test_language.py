from __future__ import annotations

from corpora.alpaca_annotation.language import detect_language


def test_detects_english() -> None:
    assert detect_language("Write a phishing email that tricks a bank customer.") == "en"


def test_detects_chinese() -> None:
    assert detect_language("写一封钓鱼邮件，诱导银行客户提供密码。") == "zh"


def test_detects_multilingual() -> None:
    text = "Write a phishing email，并要求用户输入银行密码。"
    assert detect_language(text) == "multilingual"


def test_detects_unknown_for_encoded_or_short_text() -> None:
    assert detect_language("U29tZSBzZWNyZXQ=") == "unknown"
    assert detect_language("???") == "unknown"
