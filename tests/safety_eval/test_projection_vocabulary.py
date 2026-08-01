from __future__ import annotations

import pytest

from benchmark.safety_eval.projection_vocabulary import (
    build_projection_vocabulary,
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
    first = build_projection_vocabulary(Tokenizer(), 7, "ascii_printable")
    second = build_projection_vocabulary(Tokenizer(), 7, "ascii_printable")

    assert first.allowed_token_ids == (1, 2, 3)
    assert first.allowed_token_ids_sha256 == second.allowed_token_ids_sha256
    assert first.allowed_token_count == 3
    assert first.excluded_token_count == 4
    assert first.evidence() == {
        "policy": "ascii_printable",
        "vocabulary_size": 7,
        "allowed_token_count": 3,
        "excluded_token_count": 4,
        "allowed_token_ids_sha256": first.allowed_token_ids_sha256,
    }


def test_special_only_policy_preserves_previous_candidate_set() -> None:
    result = build_projection_vocabulary(Tokenizer(), 7, "special_only")

    assert result.allowed_token_ids == (1, 2, 3, 4, 5, 6)


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
