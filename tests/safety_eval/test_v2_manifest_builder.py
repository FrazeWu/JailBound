from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from benchmark.safety_eval.manifest import AnnotationFailure
from benchmark.safety_eval.schema import FailureKind


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_safety_eval_v2_manifests.py"
V2_CONFIG = ROOT / "configs/benchmark/safety_eval_paper_v2.yaml"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_safety_eval_v2_manifests", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path, *, schema_version: str = "reviewer_eval.v2") -> Path:
    payload = OmegaConf.to_container(OmegaConf.load(V2_CONFIG), resolve=True)
    assert isinstance(payload, dict)
    payload["run"]["schema_version"] = schema_version
    payload["run"]["output_root"] = str(tmp_path / "output")
    payload["annotation"]["template_path"] = str(
        ROOT / "configs/benchmark/span_annotation_prompt.txt"
    )
    payload["annotation"]["confidence_artifact"] = str(
        tmp_path / "confidence.json"
    )
    config = tmp_path / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(payload), f=config)
    return config


def _confidence(path: Path) -> None:
    template = ROOT / "configs/benchmark/span_annotation_prompt.txt"
    path.write_text(
        json.dumps(
            {
                "artifact_version": "span_annotation_confidence.v1",
                "threshold": 0.9,
                "sources": [
                    "advbench",
                    "harmbench",
                    "safetybench",
                    "sg_bench",
                    "jailbreakbench",
                    "jailbound",
                    "s_eval",
                ],
                "annotation_provenance": [
                    {
                        "model": "de-aligned-annotator",
                        "revision": "immutable-revision",
                        "template_sha256": __import__("hashlib").sha256(
                            template.read_bytes()
                        ).hexdigest(),
                        "response_sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_builder_rejects_v1_config_before_resolving_transport(tmp_path) -> None:
    builder = _load_builder()
    config = _config(tmp_path, schema_version="reviewer_eval.v1")

    with pytest.raises(ValueError, match="reviewer_eval.v2"):
        builder.build_manifests(config, candidate_pool=25, sources=None)


def test_builder_rejects_unconfigured_repeatable_source(tmp_path) -> None:
    builder = _load_builder()
    config = _config(tmp_path)
    _confidence(tmp_path / "confidence.json")

    with pytest.raises(ValueError, match="configured source"):
        builder.build_manifests(
            config,
            candidate_pool=25,
            sources=("s_eval", "not-approved"),
        )


def test_builder_rejects_output_root_containing_legacy_manifest(tmp_path) -> None:
    builder = _load_builder()
    config = _config(tmp_path)
    _confidence(tmp_path / "confidence.json")
    legacy = tmp_path / "output/manifests/controlled_advbench.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v1 manifest"):
        builder.build_manifests(config, candidate_pool=25, sources=("s_eval",))


def test_builder_validates_frozen_annotation_provenance_before_transport(
    tmp_path,
) -> None:
    builder = _load_builder()
    config = _config(tmp_path)
    _confidence(tmp_path / "confidence.json")
    payload = json.loads((tmp_path / "confidence.json").read_text(encoding="utf-8"))
    payload["annotation_provenance"][0]["revision"] = "different"
    (tmp_path / "confidence.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        builder.build_manifests(config, candidate_pool=25, sources=("s_eval",))


def test_insufficient_builder_run_freezes_failures_before_raising(tmp_path) -> None:
    builder = _load_builder()
    failure = AnnotationFailure(
        schema_version="reviewer_eval.v2",
        example_id="s_eval:000001:fixture",
        source="s_eval",
        source_file="s_eval.jsonl",
        source_row=1,
        source_sha256="a" * 64,
        prompt_sha256="b" * 64,
        intent_sha256="c" * 64,
        failure_kind=FailureKind.annotation,
        failure_reason="SpanAnnotationError: fixture",
    )
    candidates = {"s_eval": (SimpleNamespace(prompt_sha256="d" * 64),)}

    with pytest.raises(
        ValueError,
        match=r"s_eval: eligible=1, failures=1, required=2",
    ):
        builder.audit_and_require_sufficient_candidates(
            output_root=tmp_path,
            candidates=candidates,
            failures=(failure,),
            sources=("s_eval",),
            required=2,
        )

    ledger = tmp_path / "manifests/v2/annotation_failures.jsonl"
    assert json.loads(ledger.read_text(encoding="utf-8"))["failure_kind"] == "annotation"
    assert not list((tmp_path / "manifests/v2").glob("controlled_*.jsonl"))


def test_cli_parses_repeatable_sources() -> None:
    builder = _load_builder()
    args = builder.parse_args(
        [
            "--config",
            str(V2_CONFIG),
            "--candidate-pool",
            "25",
            "--sources",
            "s_eval",
            "--sources",
            "jailbound",
        ]
    )

    assert args.sources == ["s_eval", "jailbound"]
    assert args.candidate_pool == 25


def test_transport_resolution_pins_locked_annotation_identity(monkeypatch) -> None:
    builder = _load_builder()
    observed: dict[str, object] = {}

    class FakeTransport:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    monkeypatch.setattr(builder, "OpenAIAnnotationTransport", FakeTransport)
    annotation = SimpleNamespace(
        model="annotator-family",
        revision="revision-7",
        endpoint="http://localhost:9999/v1",
    )
    builder.resolve_annotation_transport(annotation, seed=20260725, api_key="fixture")

    assert observed == {
        "model": "annotator-family",
        "revision": "revision-7",
        "endpoint": "http://localhost:9999/v1",
        "api_key": "fixture",
        "seed": 20260725,
    }
