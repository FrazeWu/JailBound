from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from benchmark.safety_eval.io import canonical_hash
from benchmark.safety_eval.projection_vocabulary import (
    COMMON_TOKEN_ID_CEILING,
    ENGLISH_LANGUAGE,
    MIN_ENGLISH_ZIPF,
    WORDFREQ_VERSION,
    PositionProjectionMask,
    build_projection_vocabulary,
    classify_projection_piece,
    validate_initial_editable_ids,
)


class Tokenizer:
    all_special_ids = (0,)
    pieces = {
        0: "<special>",
        1: " hello",
        2: ",",
        3: "\n",
        4: "\ufffd",
        5: "\u4e2d\u6587",
        6: "",
        7: "\t",
        8: "\r",
    }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_ascii_policy_is_deterministic_and_excludes_invalid_pieces() -> None:
    first = build_projection_vocabulary(Tokenizer(), 9, "ascii_printable")
    second = build_projection_vocabulary(Tokenizer(), 9, "ascii_printable")

    assert first.allowed_token_ids == (1, 2)
    assert first.allowed_token_ids_sha256 == second.allowed_token_ids_sha256
    assert first.allowed_token_count == 2
    assert first.excluded_token_count == 7
    assert first.evidence() == {
        "policy": "ascii_printable",
        "vocabulary_size": 9,
        "allowed_token_count": 2,
        "excluded_token_count": 7,
        "allowed_token_ids_sha256": first.allowed_token_ids_sha256,
    }


def test_ascii_positioned_policy_builds_exact_local_replacement_manifests() -> None:
    kwargs = {
        "z_token_ids": (1,),
        "u_token_ids": (2,),
    }

    first = build_projection_vocabulary(
        Tokenizer(),
        9,
        "ascii_printable_positioned",
        **kwargs,
    )
    second = build_projection_vocabulary(
        Tokenizer(),
        9,
        "ascii_printable_positioned",
        **kwargs,
    )

    assert first.allowed_token_ids == (1, 2)
    assert first.z_position_masks[0].allowed_token_ids == (2,)
    assert first.u_position_masks[0].allowed_token_ids == (1,)
    assert 1 not in first.z_position_masks[0].allowed_token_ids
    assert 1 in first.u_position_masks[0].allowed_token_ids
    assert 2 in first.z_position_masks[0].allowed_token_ids
    assert 2 not in first.u_position_masks[0].allowed_token_ids
    assert {
        mask.position_class
        for mask in first.z_position_masks + first.u_position_masks
    } == {"gcg_ascii_without_initial"}
    for mask in first.z_position_masks + first.u_position_masks:
        assert mask.allowed_token_ids_sha256 == canonical_hash(
            list(mask.allowed_token_ids)
        )
    assert first.position_mask_manifest_sha256 == canonical_hash(
        _full_position_manifest(first)
    )
    assert second.position_mask_manifest_sha256 == first.position_mask_manifest_sha256
    assert first.evidence() == {
        "policy": "ascii_printable_positioned",
        "vocabulary_size": 9,
        "allowed_token_count": 2,
        "excluded_token_count": 7,
        "allowed_token_ids_sha256": first.allowed_token_ids_sha256,
        "z_position_masks": [mask.evidence() for mask in first.z_position_masks],
        "u_position_masks": [mask.evidence() for mask in first.u_position_masks],
        "position_mask_manifest_sha256": first.position_mask_manifest_sha256,
    }
    assert "wordfreq_version" not in first.evidence()

    with pytest.raises(FrozenInstanceError):
        first.z_position_masks[0].position_class = "other"


@pytest.mark.parametrize(
    ("z_token_ids", "u_token_ids", "message"),
    [
        (None, (2,), "z_token_ids are required"),
        ((1,), None, "u_token_ids are required"),
        ((True,), (2,), "z_token_ids contain an invalid token ID"),
        ((1,), (9,), "u_token_ids contain an invalid token ID"),
    ],
)
def test_ascii_positioned_policy_requires_valid_initial_id_shapes(
    z_token_ids: tuple[int, ...] | None,
    u_token_ids: tuple[int, ...] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_projection_vocabulary(
            FailOnDecodeTokenizer(),
            9,
            "ascii_printable_positioned",
            z_token_ids=z_token_ids,
            u_token_ids=u_token_ids,
        )


def test_ascii_positioned_validation_fails_closed_for_invalid_manifest_state() -> None:
    result = build_projection_vocabulary(
        Tokenizer(),
        9,
        "ascii_printable_positioned",
        z_token_ids=(1,),
        u_token_ids=(2,),
    )

    validate_initial_editable_ids(result, z_token_ids=(1,), u_token_ids=(2,))

    invalid_cases = (
        (result, (), (2,)),
        (result, (0,), (2,)),
        (result, (2,), (2,)),
        (
            replace(
                result,
                z_position_masks=(
                    PositionProjectionMask(
                        original_token_id=1,
                        position_class="gcg_ascii_without_initial",
                        allowed_token_ids=(1, 2),
                        allowed_token_count=2,
                        allowed_token_ids_sha256=canonical_hash([1, 2]),
                    ),
                ),
            ),
            (1,),
            (2,),
        ),
    )
    for vocabulary, z_ids, u_ids in invalid_cases:
        with pytest.raises(ValueError, match="initial editable token"):
            validate_initial_editable_ids(
                vocabulary,
                z_token_ids=z_ids,
                u_token_ids=u_ids,
            )


def test_special_only_policy_preserves_previous_candidate_set() -> None:
    result = build_projection_vocabulary(Tokenizer(), 7, "special_only")

    assert result.allowed_token_ids == (1, 2, 3, 4, 5, 6)
    assert result.evidence() == {
        "policy": "special_only",
        "vocabulary_size": 7,
        "allowed_token_count": 6,
        "excluded_token_count": 1,
        "allowed_token_ids_sha256": result.allowed_token_ids_sha256,
    }


def test_initial_editable_ids_must_be_allowed() -> None:
    result = build_projection_vocabulary(Tokenizer(), 7, "ascii_printable")

    with pytest.raises(ValueError, match="initial editable token"):
        validate_initial_editable_ids(
            result,
            z_token_ids=(1,),
            u_token_ids=(5,),
        )


@pytest.mark.parametrize("policy", ["", "latin", "ASCII_PRINTABLE"])
def test_projection_policy_rejects_unknown_names(policy: str) -> None:
    with pytest.raises(ValueError, match="unsupported projection token policy"):
        build_projection_vocabulary(Tokenizer(), 7, policy)


@pytest.mark.parametrize(
    ("piece", "expected"),
    [
        (" word", "word_start"),
        ("For", "sentence_initial"),
        ("ative", "continuation"),
        ("'s", "contraction"),
        (",", "punctuation"),
        ("deviceId", "other"),
        (" two words", "other"),
        ("UPPER", "other"),
        ("item2", "other"),
        ("\u4e2d\u6587", "other"),
        ("", "other"),
    ],
)
def test_classify_projection_piece(piece: str, expected: str) -> None:
    assert classify_projection_piece(piece) == expected


class EnglishTokenizer:
    all_special_ids = (0,)
    pieces = {
        0: " common",
        1: " common",
        2: " obscure",
        3: " deviceId",
        4: " item2",
        5: " London",
        6: " \u4e2d\u6587",
        7: ",",
        8: "ative",
        9: "For",
        10: "'s",
        50_000: " common",
        50_001: " spare",
    }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces.get(token_id, " invalid") for token_id in token_ids)


def _frequency(word: str, language: str) -> float:
    assert language == "en"
    return {"common": 5.0, "obscure": 2.0}.get(word, 0.0)


class FailOnDecodeTokenizer:
    all_special_ids: tuple[int, ...] = ()

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        raise AssertionError("editable token validation must happen before vocabulary scanning")


class MultiCommonEnglishTokenizer(EnglishTokenizer):
    pieces = {
        **EnglishTokenizer.pieces,
        11: " beta",
        42: " alpha",
    }


def _multi_frequency(word: str, language: str) -> float:
    assert language == "en"
    return {"common": 5.0, "alpha": 4.5, "beta": 4.0}.get(word, 0.0)


def _full_position_manifest(result: object) -> dict[str, list[dict[str, object]]]:
    return {
        "z_position_masks": [
            {
                "original_token_id": mask.original_token_id,
                "position_class": mask.position_class,
                "allowed_token_ids": list(mask.allowed_token_ids),
            }
            for mask in result.z_position_masks
        ],
        "u_position_masks": [
            {
                "original_token_id": mask.original_token_id,
                "position_class": mask.position_class,
                "allowed_token_ids": list(mask.allowed_token_ids),
            }
            for mask in result.u_position_masks
        ],
    }


def test_english_common_policy_builds_canonical_position_manifest() -> None:
    result = build_projection_vocabulary(
        EnglishTokenizer(),
        50_002,
        "english_common_positioned",
        z_token_ids=(9, 1, 8),
        u_token_ids=(1, 10),
        zipf_frequency=_frequency,
        wordfreq_version="3.1.1",
    )

    assert result.common_english_token_ids == (1,)
    assert [mask.position_class for mask in result.z_position_masks] == [
        "sentence_initial",
        "word_start",
        "continuation",
    ]
    assert [mask.allowed_token_ids for mask in result.z_position_masks] == [
        (9,),
        (1,),
        (8,),
    ]
    assert [mask.allowed_token_ids for mask in result.u_position_masks] == [
        (1,),
        (10,),
    ]
    assert result.position_mask_manifest_sha256 == canonical_hash(
        _full_position_manifest(result)
    )
    assert result.evidence() == {
        "policy": "english_common_positioned",
        "vocabulary_size": 50_002,
        "allowed_token_count": 1,
        "excluded_token_count": 50_001,
        "allowed_token_ids_sha256": result.allowed_token_ids_sha256,
        "wordfreq_version": WORDFREQ_VERSION,
        "english_language": ENGLISH_LANGUAGE,
        "minimum_english_zipf": MIN_ENGLISH_ZIPF,
        "common_token_id_ceiling": COMMON_TOKEN_ID_CEILING,
        "common_english_token_count": 1,
        "common_english_token_ids_sha256": result.common_english_token_ids_sha256,
        "z_position_masks": [mask.evidence() for mask in result.z_position_masks],
        "u_position_masks": [mask.evidence() for mask in result.u_position_masks],
        "position_mask_manifest_sha256": result.position_mask_manifest_sha256,
    }


def test_word_start_original_exceptions_are_local_to_their_position() -> None:
    result = build_projection_vocabulary(
        EnglishTokenizer(),
        50_002,
        "english_common_positioned",
        z_token_ids=(2,),
        u_token_ids=(50_000,),
        zipf_frequency=_frequency,
        wordfreq_version="3.1.1",
    )

    z_mask = result.z_position_masks[0]
    u_mask = result.u_position_masks[0]
    assert z_mask.allowed_token_ids == (1, 2)
    assert u_mask.allowed_token_ids == (1, 50_000)
    assert 50_000 not in z_mask.allowed_token_ids
    assert 2 not in u_mask.allowed_token_ids

    with pytest.raises(FrozenInstanceError):
        z_mask.position_class = "other"


@pytest.mark.parametrize(
    ("z_token_ids", "u_token_ids"),
    [
        ((), (1, 2)),
        ((1, 2), ()),
        ((2,), (2,)),
    ],
)
def test_positioned_validation_preserves_block_boundaries_and_original_ids(
    z_token_ids: tuple[int, ...],
    u_token_ids: tuple[int, ...],
) -> None:
    result = build_projection_vocabulary(
        EnglishTokenizer(),
        11,
        "english_common_positioned",
        z_token_ids=(1,),
        u_token_ids=(2,),
        zipf_frequency=_frequency,
        wordfreq_version=WORDFREQ_VERSION,
    )

    with pytest.raises(ValueError, match="initial editable token"):
        validate_initial_editable_ids(
            result,
            z_token_ids=z_token_ids,
            u_token_ids=u_token_ids,
        )


@pytest.mark.parametrize(
    ("z_token_ids", "u_token_ids", "message"),
    [
        (None, (1,), "z_token_ids are required"),
        ((1,), None, "u_token_ids are required"),
        ((True,), (1,), "z_token_ids contain an invalid token ID"),
        ((1,), (3,), "u_token_ids contain an invalid token ID"),
    ],
)
def test_positioned_editable_ids_fail_before_common_vocabulary_scan(
    z_token_ids: tuple[int, ...] | None,
    u_token_ids: tuple[int, ...] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_projection_vocabulary(
            FailOnDecodeTokenizer(),
            3,
            "english_common_positioned",
            z_token_ids=z_token_ids,
            u_token_ids=u_token_ids,
            zipf_frequency=_frequency,
            wordfreq_version=WORDFREQ_VERSION,
        )


def test_multiple_common_candidates_are_sorted_with_deterministic_digest() -> None:
    kwargs = {
        "z_token_ids": (1,),
        "u_token_ids": (42,),
        "zipf_frequency": _multi_frequency,
        "wordfreq_version": WORDFREQ_VERSION,
    }

    first = build_projection_vocabulary(
        MultiCommonEnglishTokenizer(),
        43,
        "english_common_positioned",
        **kwargs,
    )
    second = build_projection_vocabulary(
        MultiCommonEnglishTokenizer(),
        43,
        "english_common_positioned",
        **kwargs,
    )

    assert first.common_english_token_ids == (1, 11, 42)
    assert first.common_english_token_ids_sha256 == canonical_hash([1, 11, 42])
    assert second.common_english_token_ids_sha256 == first.common_english_token_ids_sha256


@pytest.mark.parametrize("wordfreq_version", ["3.0.0", ""])
def test_english_common_policy_rejects_wrong_wordfreq_version(
    wordfreq_version: str,
) -> None:
    with pytest.raises(RuntimeError, match="wordfreq version"):
        build_projection_vocabulary(
            EnglishTokenizer(),
            50_002,
            "english_common_positioned",
            z_token_ids=(1,),
            u_token_ids=(1,),
            zipf_frequency=_frequency,
            wordfreq_version=wordfreq_version,
        )
