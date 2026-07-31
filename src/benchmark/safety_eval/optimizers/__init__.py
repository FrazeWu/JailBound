"""Approved optimizer identities and their constructors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any


OptimizerFactory = Callable[..., object]


def _build_init() -> object:
    from .jailbound import InitOptimizer

    return InitOptimizer()


def _build_jailbound(
    method: str,
    *,
    learning_rate: float = 0.01,
    max_grad_norm: float = 1.0,
) -> object:
    from .jailbound import build_jailbound_optimizer

    return build_jailbound_optimizer(
        method,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
    )


def _build_pez(*args: Any, **kwargs: Any) -> object:
    from .pez import PEZOptimizer

    return PEZOptimizer(*args, **kwargs)


def _build_gbda(*args: Any, **kwargs: Any) -> object:
    from .gbda import GBDAOptimizer

    return GBDAOptimizer(*args, **kwargs)


def _build_gbda_official(*args: Any, **kwargs: Any) -> object:
    from .gbda_official import OfficialGBDAOptimizer

    return OfficialGBDAOptimizer(*args, **kwargs)


def _build_gcg(*args: Any, **kwargs: Any) -> object:
    from .gcg import GCGOptimizer

    return GCGOptimizer(*args, **kwargs)


def _build_dual_branch(*args: Any, **kwargs: Any) -> object:
    from .jailbound import DualBranchOptimizer

    return DualBranchOptimizer(*args, **kwargs)


def _build_random_mutation() -> object:
    from .random_mutation import RandomMutationOptimizer

    return RandomMutationOptimizer()


OPTIMIZER_REGISTRY: dict[str, OptimizerFactory] = {
    "init": _build_init,
    "random_mutation": _build_random_mutation,
    "zol": partial(_build_jailbound, "zol"),
    "pez": _build_pez,
    "gbda": _build_gbda,
    "gbda_official": _build_gbda_official,
    "gcg": _build_gcg,
    "jailbound_o_minus": partial(_build_jailbound, "jailbound_o_minus"),
    "jailbound_o_plus": partial(_build_jailbound, "jailbound_o_plus"),
    "dual_branch": _build_dual_branch,
}


def validate_optimizer_methods(methods: Sequence[str]) -> None:
    """Require the configured methods to match the approved identities and order."""
    if list(methods) != list(OPTIMIZER_REGISTRY):
        raise ValueError("configured methods must equal the approved optimizer methods exactly once")
