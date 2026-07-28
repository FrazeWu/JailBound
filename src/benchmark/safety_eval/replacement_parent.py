"""Assemble a provenance-preserving parent view after one source replacement."""

from __future__ import annotations

from pathlib import Path

from .io import atomic_write_json


SELECTED_SOURCES = ("harmbench", "jailbound", "s_eval")
REPLACED_SOURCE = "jailbound"


def _required_artifacts(*, root: Path, source: str, target_key: str, primary_judge: str) -> tuple[Path, ...]:
    return (
        root / "manifests" / f"controlled_{source}.jsonl",
        root / "optimization" / source,
        root / "responses" / target_key / source,
        root / "judgments" / primary_judge / target_key / source,
    )


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def assemble_replacement_parent(
    *,
    original_root: str | Path,
    replacement_root: str | Path,
    output_root: str | Path,
    target_key: str,
    primary_judge: str,
) -> Path:
    """Create an immutable three-source view with only JailBound replaced.

    The function creates links rather than copying artifacts, and refuses to
    overwrite a pre-existing output path. All required artifacts are validated
    before the first link is created.
    """

    original = Path(original_root).resolve()
    replacement = Path(replacement_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("replacement parent output already exists")
    if not target_key or not primary_judge:
        raise ValueError("target key and primary judge are required")
    source_roots = {
        "harmbench": original,
        REPLACED_SOURCE: replacement,
        "s_eval": original,
    }
    for source, root in source_roots.items():
        for artifact in _required_artifacts(
            root=root,
            source=source,
            target_key=target_key,
            primary_judge=primary_judge,
        ):
            if not artifact.exists():
                raise ValueError(f"required artifact is missing: {source}/{artifact.name}")

    for source in SELECTED_SOURCES:
        root = source_roots[source]
        _link(
            root / "manifests" / f"controlled_{source}.jsonl",
            output / "manifests" / f"controlled_{source}.jsonl",
        )
        _link(root / "optimization" / source, output / "optimization" / source)
        _link(
            root / "responses" / target_key / source,
            output / "responses" / target_key / source,
        )
        _link(
            root / "judgments" / primary_judge / target_key / source,
            output / "judgments" / primary_judge / target_key / source,
        )
    atomic_write_json(
        output / "replacement_provenance.json",
        {
            "original_root": str(original),
            "replaced_source": REPLACED_SOURCE,
            "replacement_root": str(replacement),
            "sources": {source: str(source_roots[source]) for source in SELECTED_SOURCES},
            "target_key": target_key,
            "primary_judge": primary_judge,
        },
    )
    return output


def seed_reused_secondary_judgments(
    *,
    original_selection_root: str | Path,
    output_root: str | Path,
    target_key: str,
    secondary_judge: str = "qwen32_compat",
) -> Path:
    """Seed a new gate root with secondary judgments for unchanged sources only."""

    original = Path(original_selection_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("replacement selection output already exists")
    if not target_key or not secondary_judge:
        raise ValueError("target key and secondary judge are required")
    unchanged_sources = tuple(source for source in SELECTED_SOURCES if source != REPLACED_SOURCE)
    targets = {
        source: original / "judgments" / secondary_judge / target_key / source
        for source in unchanged_sources
    }
    for source, target in targets.items():
        if not target.is_dir():
            raise ValueError(f"required reused secondary judgment is missing: {source}")
    for source, target in targets.items():
        _link(target, output / "judgments" / secondary_judge / target_key / source)
    atomic_write_json(
        output / "reused_secondary_provenance.json",
        {
            "original_selection_root": str(original),
            "replaced_source": REPLACED_SOURCE,
            "reused_sources": list(unchanged_sources),
            "secondary_judge": secondary_judge,
            "target_key": target_key,
        },
    )
    return output
