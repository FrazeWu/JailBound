from __future__ import annotations

import json

import pytest
import torch

from benchmark.safety_eval.paper_v2_gcg import (
    audit_gcg_changes,
    build_gcg_initial_state,
    reconstruct_gcg_token_ids,
    standard_gcg_forbidden_token_ids,
)
from benchmark.safety_eval.projection_vocabulary import build_projection_vocabulary
from benchmark.safety_eval.prompt_contract import TokenizedEditablePrompt


class _WrongWidthPrompt(TokenizedEditablePrompt):
    def gather_editable_ids(self) -> torch.Tensor:
        return self.base_token_ids[:, :1]


def _prompt(
    *,
    base_token_ids: torch.Tensor | None = None,
    prompt_type: type[TokenizedEditablePrompt] = TokenizedEditablePrompt,
) -> TokenizedEditablePrompt:
    base = torch.tensor([[10, 11, 12, 13, 14]]) if base_token_ids is None else base_token_ids
    return prompt_type(
        full_text="abcde",
        base_token_ids=base,
        attention_mask=torch.ones_like(base, dtype=torch.long),
        editable_positions=(1, 3),
        frozen_positions=(0, 2, 4),
        token_offsets=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
        boundary_expansions=((1, 2), (3, 4)),
        tokenizer_revision="fixture-r1",
    )


def test_build_gcg_initial_state_uses_readable_prefix_and_annotated_u() -> None:
    prefix_ids = torch.tensor([[21, 22]])

    state = build_gcg_initial_state(_prompt(), prefix_ids=prefix_ids)

    assert torch.equal(state.z, prefix_ids)
    assert state.u.tolist() == [[11, 13]]
    assert state.u.tolist() != [[13, 14]]
    assert torch.equal(state.z0, state.z)
    assert torch.equal(state.u0, state.u)
    assert state.z0.data_ptr() != state.z.data_ptr()
    assert state.u0.data_ptr() != state.u.data_ptr()
    assert state.z0.requires_grad is False
    assert state.u0.requires_grad is False


def test_reconstruct_gcg_token_ids_prepends_z_and_preserves_frozen_ids() -> None:
    prompt = _prompt()

    full = reconstruct_gcg_token_ids(
        prompt,
        z_token_ids=torch.tensor([[21, 22]]),
        u_token_ids=torch.tensor([[31, 32]]),
    )

    assert full.tolist() == [[21, 22, 10, 31, 12, 32, 14]]
    reconstructed_prompt = full[:, 2:]
    assert torch.equal(
        reconstructed_prompt[:, prompt.frozen_positions],
        prompt.base_token_ids[:, prompt.frozen_positions],
    )


def test_audit_gcg_changes_counts_unique_positions_and_is_json_serializable() -> None:
    audit = audit_gcg_changes(
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[4, 5]]),
        torch.tensor([[9, 2, 8]]),
        torch.tensor([[4, 7]]),
    )

    assert audit == {
        "z": 2,
        "u": 1,
        "total": 3,
        "z_positions": [0, 2],
        "u_positions": [1],
    }
    assert json.loads(json.dumps(audit)) == audit


@pytest.mark.parametrize(
    "prefix_ids",
    [
        torch.tensor([21, 22]),
        torch.tensor([[21.0, 22.0]]),
        torch.tensor([[21.0 + 0.0j, 22.0 + 0.0j]]),
        torch.tensor([[True, False]]),
        torch.empty((1, 0), dtype=torch.long),
        torch.tensor([[21, 22], [23, 24]]),
    ],
)
def test_build_gcg_initial_state_rejects_invalid_prefix_ids(prefix_ids: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        build_gcg_initial_state(_prompt(), prefix_ids=prefix_ids)


def test_build_gcg_initial_state_rejects_non_tensor_prefix_ids() -> None:
    with pytest.raises(ValueError, match="tensor"):
        build_gcg_initial_state(_prompt(), prefix_ids=[21, 22])  # type: ignore[arg-type]


def test_build_gcg_initial_state_rejects_non_integer_annotated_ids() -> None:
    with pytest.raises(ValueError, match="integer"):
        build_gcg_initial_state(
            _prompt(base_token_ids=torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]])),
            prefix_ids=torch.tensor([[21, 22]]),
        )


def test_build_gcg_initial_state_requires_exact_omega_s_width() -> None:
    with pytest.raises(ValueError, match="Omega_s"):
        build_gcg_initial_state(
            _prompt(prompt_type=_WrongWidthPrompt),
            prefix_ids=torch.tensor([[21, 22]]),
        )


def test_standard_gcg_forbidden_tokens_exclude_non_ascii_and_non_printable() -> None:
    class Tokenizer:
        all_special_ids = [4]

        @staticmethod
        def decode(ids: list[int], **_: object) -> str:
            return {
                0: " word",
                1: "例",
                2: "\n",
                3: "!",
                4: "<|special|>",
            }[ids[0]]

    forbidden = standard_gcg_forbidden_token_ids(Tokenizer(), vocabulary_size=5)

    assert forbidden == (1, 2, 4)


def test_ascii_projection_is_exact_complement_of_standard_gcg_filter() -> None:
    class Tokenizer:
        all_special_ids = [0]

        @staticmethod
        def decode(
            ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return {
                0: "<special>",
                1: " word",
                2: "!",
                3: "\n",
                4: "\t",
                5: "\r",
                6: "\ufffd",
                7: "\u4f8b",
                8: "",
            }[ids[0]]

    tokenizer = Tokenizer()
    vocabulary = build_projection_vocabulary(tokenizer, 9, "ascii_printable")
    forbidden = standard_gcg_forbidden_token_ids(tokenizer, vocabulary_size=9)

    assert vocabulary.allowed_token_ids == (1, 2)
    assert forbidden == (0, 3, 4, 5, 6, 7, 8)
    assert vocabulary.allowed_token_ids == tuple(
        token_id for token_id in range(9) if token_id not in forbidden
    )


@pytest.mark.parametrize(
    ("z_token_ids", "u_token_ids"),
    [
        (torch.tensor([21, 22]), torch.tensor([[31, 32]])),
        (torch.tensor([[21.0, 22.0]]), torch.tensor([[31, 32]])),
        (torch.tensor([[21.0 + 0.0j, 22.0 + 0.0j]]), torch.tensor([[31, 32]])),
        (torch.tensor([[21, 22]]), torch.tensor([[[31, 32]]])),
        (torch.tensor([[21, 22]]), torch.tensor([[True, False]])),
        (torch.empty((1, 0), dtype=torch.long), torch.tensor([[31, 32]])),
        (torch.tensor([[21, 22]]), torch.tensor([[31]])),
        (torch.tensor([[21, 22], [23, 24]]), torch.tensor([[31, 32]])),
    ],
)
def test_reconstruct_gcg_token_ids_rejects_invalid_ids(
    z_token_ids: torch.Tensor,
    u_token_ids: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        reconstruct_gcg_token_ids(
            _prompt(),
            z_token_ids=z_token_ids,
            u_token_ids=u_token_ids,
        )


def test_reconstruct_gcg_token_ids_rejects_non_tensor_ids() -> None:
    with pytest.raises(ValueError, match="tensor"):
        reconstruct_gcg_token_ids(
            _prompt(),
            z_token_ids=torch.tensor([[21, 22]]),
            u_token_ids=[31, 32],  # type: ignore[arg-type]
        )


def test_reconstruct_gcg_token_ids_rejects_non_integer_base_ids() -> None:
    with pytest.raises(ValueError, match="integer"):
        reconstruct_gcg_token_ids(
            _prompt(base_token_ids=torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]])),
            z_token_ids=torch.tensor([[21, 22]]),
            u_token_ids=torch.tensor([[31, 32]]),
        )


@pytest.mark.parametrize(
    ("initial_z", "initial_u", "current_z", "current_u"),
    [
        (
            torch.tensor([[1, 2]]),
            torch.tensor([[3, 4]]),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[3, 4]]),
        ),
        (
            torch.tensor([[1, 2]]),
            torch.tensor([[3, 4]]),
            torch.tensor([[1, 2]]),
            torch.tensor([[3.0, 4.0]]),
        ),
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
            torch.tensor([[1, 9], [3, 4]]),
            torch.tensor([[5, 6], [7, 0]]),
        ),
    ],
)
def test_audit_gcg_changes_rejects_invalid_or_ambiguous_inputs(
    initial_z: torch.Tensor,
    initial_u: torch.Tensor,
    current_z: torch.Tensor,
    current_u: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        audit_gcg_changes(initial_z, initial_u, current_z, current_u)


def test_audit_gcg_changes_rejects_non_tensor_ids() -> None:
    with pytest.raises(ValueError, match="tensor"):
        audit_gcg_changes(
            torch.tensor([[1, 2]]),
            torch.tensor([[3, 4]]),
            [1, 2],  # type: ignore[arg-type]
            torch.tensor([[3, 4]]),
        )
