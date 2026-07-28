from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.runner import OptimizationJob, OptimizationRunner, OptimizationSnapshot, RunConfigMismatch
from benchmark.safety_eval.schema import ComputeCounters


@dataclass
class DeterministicFakeOptimizer:
    fail_after: int | None = None

    def __post_init__(self) -> None:
        self.requests: list[tuple[int, ...]] = []

    def snapshots(self, checkpoints: tuple[int, ...]) -> Iterable[OptimizationSnapshot]:
        self.requests.append(checkpoints)
        for index, checkpoint in enumerate(checkpoints):
            if self.fail_after is not None and index == self.fail_after:
                raise RuntimeError("interrupted fixture run")
            yield OptimizationSnapshot(
                checkpoint=checkpoint,
                representation="fixture_ids",
                attack_loss=float(checkpoint),
                counters=ComputeCounters(updates=checkpoint, forward_passes=checkpoint + 1),
            )


def _job() -> OptimizationJob:
    return OptimizationJob(
        source="fixture_source",
        method="fixture_method",
        cell_id="cell:fixture",
        sample_id="sample:fixture",
        random_seed=17,
    )


def test_runner_resumes_interrupted_checkpoint_records_without_duplicates(tmp_path) -> None:
    runner = OptimizationRunner(
        tmp_path,
        config_hash="a" * 64,
        run_id="run:fixture",
        git_revision="fixture-revision",
    )
    interrupted = DeterministicFakeOptimizer(fail_after=1)

    with pytest.raises(RuntimeError, match="interrupted fixture run"):
        runner.run(_job(), checkpoints=(0, 1, 2), snapshot_factory=interrupted.snapshots)

    resumed = DeterministicFakeOptimizer()
    records = runner.run(_job(), checkpoints=(0, 1, 2), snapshot_factory=resumed.snapshots)
    path = tmp_path / "optimization" / "fixture_source" / "fixture_method" / "records.jsonl"

    assert interrupted.requests == [(0, 1, 2)]
    assert resumed.requests == [(1, 2)]
    assert [record.checkpoint for record in records] == [1, 2]
    assert [row["checkpoint"] for row in read_jsonl(path)] == [0, 1, 2]
    assert len({(row["cell_id"], row["sample_id"], row["checkpoint"]) for row in read_jsonl(path)}) == 3
    assert runner.run(_job(), checkpoints=(0, 1, 2), snapshot_factory=resumed.snapshots) == []
    assert resumed.requests == [(1, 2)]


def test_runner_rejects_a_different_config_hash_for_the_same_output_root(tmp_path) -> None:
    OptimizationRunner(tmp_path, config_hash="a" * 64, run_id="run:fixture", git_revision="fixture-revision")

    with pytest.raises(RunConfigMismatch, match="different config hash"):
        OptimizationRunner(tmp_path, config_hash="b" * 64, run_id="run:fixture", git_revision="fixture-revision")
