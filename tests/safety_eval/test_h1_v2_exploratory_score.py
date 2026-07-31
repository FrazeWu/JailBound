from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "analyze_h1_v2_exploratory_score_test",
        ROOT / "scripts" / "analyze_h1_v2_exploratory_score.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exploratory_markdown_cannot_be_mistaken_for_the_confirmatory_result() -> None:
    module = _module()

    rendered = module._render_markdown(
        {
            "source_rho": {"jailbound": 0.3, "s_eval": 0.2},
            "mean_rho": 0.25,
            "permutation_pvalue": 0.03,
            "bootstrap_95_ci": [0.01, 0.4],
            "endpoint_bands_descriptive_only": [],
        }
    )

    assert "post-hoc exploratory" in rendered
    assert "does not replace" in rendered
    assert "| JailBound |" in rendered
