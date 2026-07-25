from __future__ import annotations

from corpora.alpaca_annotation.taxonomy import (
    ATTACK_TYPE_NAMES,
    DOMAIN_NAMES,
    RISK_CATEGORY_NAMES,
    build_instruction,
    normalize_annotation_label,
)


def test_vocabularies_include_expected_labels() -> None:
    assert "Persuasion & Deception" in ATTACK_TYPE_NAMES
    assert "Cybersecurity Misuse" in RISK_CATEGORY_NAMES
    assert "Developer Tools / Code" in DOMAIN_NAMES
    assert "Security Operations / Cybersecurity" in DOMAIN_NAMES
    assert "Security Operations / CyberSecurity" not in DOMAIN_NAMES


def test_normalize_annotation_label_falls_back_to_default() -> None:
    assert normalize_annotation_label("unknown", ATTACK_TYPE_NAMES, "Prefix Injection") == "Prefix Injection"
    assert normalize_annotation_label("cybersecurity misuse", RISK_CATEGORY_NAMES, "Unsafe or Unethical Behavioral Encouragement") == "Cybersecurity Misuse"


def test_normalize_annotation_label_handles_vllm_punctuation() -> None:
    assert normalize_annotation_label("Cybersecurity Misuse.", RISK_CATEGORY_NAMES, "Unsafe or Unethical Behavioral Encouragement") == "Cybersecurity Misuse"


def test_normalize_annotation_label_handles_colon_prefixed_output() -> None:
    assert normalize_annotation_label("risk_category: Cybersecurity Misuse", RISK_CATEGORY_NAMES, "Unsafe or Unethical Behavioral Encouragement") == "Cybersecurity Misuse"


def test_normalize_annotation_label_handles_numbered_quoted_output() -> None:
    assert normalize_annotation_label('1. "Developer Tools / Code"', DOMAIN_NAMES, "Education") == "Developer Tools / Code"


def test_normalize_annotation_label_handles_single_word_colon_prefixed_output() -> None:
    assert normalize_annotation_label("domain: Education", DOMAIN_NAMES, "Developer Tools / Code") == "Education"


def test_normalize_annotation_label_handles_single_word_numbered_quoted_output() -> None:
    assert normalize_annotation_label('1. "Education"', DOMAIN_NAMES, "Developer Tools / Code") == "Education"


def test_build_instruction_mentions_all_dimensions() -> None:
    instruction = build_instruction(
        "Persuasion & Deception",
        "Cybersecurity Misuse",
        "Developer Tools / Code",
    )

    assert "Persuasion & Deception" in instruction
    assert "Cybersecurity Misuse" in instruction
    assert "Developer Tools / Code" in instruction
    assert "jailbreak attack prompt" in instruction
