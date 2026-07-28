from __future__ import annotations

import pytest

from benchmark.safety_eval.optimizers import (
    OPTIMIZER_REGISTRY,
    validate_optimizer_methods,
)
from benchmark.safety_eval.optimizers.random_mutation import RandomMutationOptimizer


APPROVED_METHODS = [
    "init",
    "random_mutation",
    "zol",
    "pez",
    "gbda",
    "gcg",
    "jailbound_o_minus",
    "jailbound_o_plus",
    "dual_branch",
]


def test_registry_contains_only_approved_methods_in_order() -> None:
    assert list(OPTIMIZER_REGISTRY) == APPROVED_METHODS


def test_validate_optimizer_methods_accepts_the_exact_approved_sequence() -> None:
    validate_optimizer_methods(APPROVED_METHODS)


@pytest.mark.parametrize(
    "methods",
    [
        APPROVED_METHODS[:-1],
        [*APPROVED_METHODS[:-1], "zol"],
        ["init", "random_mutation", "no_fol", *APPROVED_METHODS[3:]],
        list(reversed(APPROVED_METHODS)),
    ],
)
def test_validate_optimizer_methods_rejects_missing_duplicate_alias_and_reordered_methods(
    methods: list[str],
) -> None:
    with pytest.raises(ValueError, match="approved optimizer methods"):
        validate_optimizer_methods(methods)


def test_random_mutation_uses_its_concrete_factory() -> None:
    assert isinstance(OPTIMIZER_REGISTRY["random_mutation"](), RandomMutationOptimizer)
