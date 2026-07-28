"""Differentiable dual-anchor objective primitives for editable attack states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class EditableState:
    z: torch.Tensor
    u: torch.Tensor
    z0: torch.Tensor
    u0: torch.Tensor


@dataclass(frozen=True)
class ObjectiveValue:
    maximize: torch.Tensor
    attack_loss: torch.Tensor
    proxy_risk: torch.Tensor
    fol: torch.Tensor | None
    answer_logp: torch.Tensor
    refusal_logp: torch.Tensor
    margin: torch.Tensor


class AttackObjective:
    """Minimal model-agnostic core; model adapters supply anchor projections."""

    def __init__(self, answer_vector: torch.Tensor, refusal_vector: torch.Tensor, *, epsilon: float, lambda_fol: float, gamma_z: float, gamma_u: float) -> None:
        self.answer_vector = answer_vector.detach().clone()
        self.refusal_vector = refusal_vector.detach().clone()
        self.epsilon, self.lambda_fol, self.gamma_z, self.gamma_u = epsilon, lambda_fol, gamma_z, gamma_u

    def _scores(self, state: EditableState) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.cat((state.z, state.u), dim=1).mean(dim=1)
        answer = (pooled * self.answer_vector.to(pooled)).sum(dim=-1).mean()
        refusal = (pooled * self.refusal_vector.to(pooled)).sum(dim=-1).mean()
        return answer, refusal

    def evaluate(self, state: EditableState, *, fol_sign: Literal[-1, 0, 1] = 0, include_fol: bool = False) -> ObjectiveValue:
        answer, refusal = self._scores(state)
        proxy = answer - refusal
        attack = proxy - self.gamma_z * (state.z - state.z0).square().sum() - self.gamma_u * (state.u - state.u0).square().sum()
        fol = None
        maximize = attack
        if include_fol:
            gradients = torch.autograd.grad(attack, (state.z, state.u), create_graph=True, retain_graph=True)
            fol = self.epsilon * torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
            maximize = attack + fol_sign * self.lambda_fol * fol
        return ObjectiveValue(maximize, attack, proxy, fol, answer, refusal, proxy)

    def hvp(self, state: EditableState, direction: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loss = self.evaluate(state).attack_loss
        gradient = torch.autograd.grad(loss, (state.z, state.u), create_graph=True)
        directional = sum((left * right).sum() for left, right in zip(gradient, direction))
        return torch.autograd.grad(directional, (state.z, state.u))
