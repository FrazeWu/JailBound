from __future__ import annotations

import torch
from torch import nn

from benchmark.safety_eval.paper_v2_ablation import build_paper_v2_continuous_chat_input
from benchmark.safety_eval.prompt_contract import TokenizedEditablePrompt


class _CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False and add_generation_prompt is True
        return "<s>" + messages[-1]["content"] + "</s>"

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs == {
            "return_tensors": "pt",
            "add_special_tokens": False,
            "return_offsets_mapping": True,
        }
        return {
            "input_ids": torch.tensor([[ord(character) for character in text]]),
            "offset_mapping": torch.tensor([[[index, index + 1] for index in range(len(text))]]),
        }


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 3)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


def test_continuous_chat_input_scatters_u_and_prepends_z_inside_user_content() -> None:
    tokenizer = _CharacterTokenizer()
    model = _Model()
    base_ids = torch.tensor([[ord(character) for character in "abcd"]])
    prompt = TokenizedEditablePrompt(
        full_text="abcd",
        base_token_ids=base_ids,
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        editable_positions=(1, 3),
        frozen_positions=(0, 2),
        token_offsets=((0, 1), (1, 2), (2, 3), (3, 4)),
        boundary_expansions=((1, 2), (3, 4)),
        tokenizer_revision="fixture-r1",
    )
    z = torch.tensor([[[7.0, 8.0, 9.0]]])
    u = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    result = build_paper_v2_continuous_chat_input(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        z=z,
        u=u,
    )

    rendered_ids = torch.tensor([[ord(character) for character in "<s>abcd</s>"]])
    rendered_embeds = model.embedding(rendered_ids).detach()
    expected = torch.cat(
        (
            rendered_embeds[:, :3],
            z,
            rendered_embeds[:, 3:4],
            u[:, :1],
            rendered_embeds[:, 5:6],
            u[:, 1:],
            rendered_embeds[:, 7:],
        ),
        dim=1,
    )
    assert torch.allclose(result.inputs_embeds, expected)
    assert result.attention_mask.tolist() == [[1] * expected.shape[1]]
    assert result.base_content_token_ids == tuple(base_ids[0].tolist())
