from __future__ import annotations

from contextlib import contextmanager
import hashlib
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


def test_build_input_snapshot_keeps_hash_and_consumed_bytes_identical(tmp_path) -> None:
    builder = _load_builder()
    originals = {
        "config": tmp_path / "config.yaml",
        "confidence_artifact": tmp_path / "confidence.json",
        "annotation_template": tmp_path / "annotation.txt",
        "taxonomy_mapping": tmp_path / "taxonomy.yaml",
        "harmbench_targets": tmp_path / "targets.json",
        "source:s_eval": tmp_path / "S-Eval_attack_en_full.jsonl",
    }
    original_bytes = {
        name: f"version-a:{name}\n".encode() for name in originals
    }
    for name, path in originals.items():
        path.write_bytes(original_bytes[name])

    with builder.snapshot_build_inputs(originals) as snapshot:
        snapshot_paths = dict(snapshot.paths)
        for name, path in originals.items():
            path.write_bytes(f"version-b:{name}\n".encode())
            assert snapshot.paths[name].read_bytes() == original_bytes[name]
            assert snapshot.hashes[name] == hashlib.sha256(
                original_bytes[name]
            ).hexdigest()
        assert snapshot.paths["source:s_eval"].name == (
            "S-Eval_attack_en_full.jsonl"
        )

    assert all(not path.exists() for path in snapshot_paths.values())


def test_builder_loads_s_eval_from_hashed_snapshot_after_source_replacement(
    tmp_path, monkeypatch
) -> None:
    builder = _load_builder()
    source = tmp_path / "S-Eval_attack_en_full.jsonl"
    original_bytes = b'{"prompt":"version-a"}\n'
    source.write_bytes(original_bytes)
    config_path = _config(tmp_path)
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(payload, dict)
    payload["data"]["paths"]["s_eval"] = str(source)
    OmegaConf.save(config=OmegaConf.create(payload), f=config_path)
    _confidence(tmp_path / "confidence.json")

    original_snapshotter = builder.snapshot_build_inputs

    @contextmanager
    def replace_after_snapshot(paths):
        with original_snapshotter(paths) as snapshot:
            if "source:s_eval" in paths:
                source.write_bytes(b'{"prompt":"version-b"}\n')
            yield snapshot

    observed: dict[str, object] = {}

    class ObservedSnapshot(RuntimeError):
        pass

    def observe_source(source_name, path, targets_path):
        observed.update(
            source=source_name,
            path=path,
            bytes=Path(path).read_bytes(),
            targets_path=targets_path,
        )
        return (
            [
                builder.RawExample(
                    source="s_eval",
                    source_row=0,
                    intent="fixture intent",
                    attack_text="fixture prompt",
                    target_text=None,
                    source_risk_label="risk",
                    source_attack_label="direct_request",
                    source_domain_label=None,
                    language="en",
                    preprocessing=(),
                )
            ],
            SimpleNamespace(raw_count=1, eligible_count=1, exclusions={}),
        )

    def observe_annotation(*args, **kwargs):
        observed["source_file"] = kwargs["source_file"]
        raise ObservedSnapshot

    class FakeEncoder:
        resolved_revision = "fixture-encoder"

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            return [[0.0] for _ in texts]

    monkeypatch.setattr(builder, "snapshot_build_inputs", replace_after_snapshot)
    monkeypatch.setattr(builder, "load_source_with_report", observe_source)
    monkeypatch.setattr(builder, "annotate_raw_candidates", observe_annotation)
    monkeypatch.setattr(
        builder,
        "load_taxonomy_mapping",
        lambda path: {
            "risk_categories": {"risk": {"description": "risk"}},
            "threat_domains": {"domain": {"description": "domain"}},
        },
    )
    monkeypatch.setattr(builder, "QwenHiddenMeanEncoder", FakeEncoder)
    monkeypatch.setattr(builder, "SpanAnnotator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        builder, "resolve_annotation_transport", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        builder,
        "_lock_v2_runtime_config",
        lambda *args, **kwargs: SimpleNamespace(
            run_id="run:fixture", config_hash="a" * 64
        ),
    )

    with pytest.raises(ObservedSnapshot):
        builder.build_manifests(
            config_path,
            candidate_pool=25,
            sources=("s_eval",),
        )

    snapshot_path = observed["path"]
    assert observed["source"] == "s_eval"
    assert observed["bytes"] == original_bytes
    assert Path(snapshot_path).name == source.name
    assert Path(snapshot_path) != source
    assert observed["source_file"] == str(source)
    assert not Path(snapshot_path).exists()


@pytest.mark.parametrize(
    "changed_input", ("confidence_artifact", "harmbench_targets")
)
def test_runtime_lock_rejects_changed_build_input_bytes(
    tmp_path, changed_input: str
) -> None:
    builder = _load_builder()
    config_path = _config(tmp_path)
    config = builder.load_v2_config(config_path)
    confidence = tmp_path / "confidence.json"
    targets = tmp_path / "harmbench_targets.json"
    confidence.write_bytes(b'{"threshold":0.9}\n')
    targets.write_bytes(b'{"target":"fixture"}\n')
    input_paths = {
        "confidence_artifact": confidence,
        "harmbench_targets": targets,
    }

    def input_hashes() -> dict[str, str]:
        return {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in input_paths.items()
        }

    output_root = tmp_path / "runtime"
    locked = builder._lock_v2_runtime_config(
        config,
        output_root=output_root,
        source_hashes={"s_eval": "a" * 64},
        build_input_hashes=input_hashes(),
    )
    run_manifest = json.loads(
        (output_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["build_input_hashes"] == input_hashes()
    assert locked.run_id == run_manifest["run_id"]

    input_paths[changed_input].write_bytes(
        input_paths[changed_input].read_bytes() + b" "
    )
    with pytest.raises(ValueError, match="runtime lock differs"):
        builder._lock_v2_runtime_config(
            config,
            output_root=output_root,
            source_hashes={"s_eval": "a" * 64},
            build_input_hashes=input_hashes(),
        )


def test_insufficient_builder_run_freezes_failures_before_raising(tmp_path) -> None:
    builder = _load_builder()
    failures = (
        AnnotationFailure(
            schema_version="reviewer_eval.v2",
            example_id="s_eval:000001:fixture",
            source="s_eval",
            source_file="s_eval.jsonl",
            source_row=1,
            source_sha256="a" * 64,
            prompt_sha256="b" * 64,
            intent_sha256="c" * 64,
            failure_kind=FailureKind.annotation,
            failure_reason="annotation_error",
        ),
        AnnotationFailure(
            schema_version="reviewer_eval.v2",
            example_id="jailbound:000001:fixture",
            source="jailbound",
            source_file="jailbound.json",
            source_row=1,
            source_sha256="d" * 64,
            prompt_sha256="e" * 64,
            intent_sha256="f" * 64,
            failure_kind=FailureKind.transport,
            failure_reason="transport_error",
        ),
    )
    candidates = {
        "s_eval": (SimpleNamespace(prompt_sha256="d" * 64),),
        "jailbound": (SimpleNamespace(prompt_sha256="e" * 64),),
    }

    with pytest.raises(
        ValueError,
        match=r"s_eval: eligible=1, failures=1, required=2",
    ):
        builder.audit_and_require_sufficient_candidates(
            output_root=tmp_path,
            candidates=candidates,
            failures=failures,
            sources=("s_eval", "jailbound"),
            required=2,
            report={"build_input_hashes": {"confidence_artifact": "9" * 64}},
        )

    ledger = tmp_path / "manifests/v2/annotation_failures.jsonl"
    assert {
        json.loads(line)["failure_kind"]
        for line in ledger.read_text(encoding="utf-8").splitlines()
    } == {"annotation", "transport"}
    report = json.loads(
        (tmp_path / "manifests/v2/source_ingestion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "insufficient_candidates"
    assert report["build_input_hashes"] == {"confidence_artifact": "9" * 64}
    assert report["required_counts_by_source"] == {"jailbound": 2, "s_eval": 2}
    assert report["eligible_counts_by_source"] == {"jailbound": 1, "s_eval": 1}
    assert report["failure_counts_by_source_and_kind"] == {
        "jailbound": {"transport": 1},
        "s_eval": {"annotation": 1},
    }
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
