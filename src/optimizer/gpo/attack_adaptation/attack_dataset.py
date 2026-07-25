"""
AttackDataset — wraps HarmBench behaviors for GPO-compatible iteration.

Loads behaviors from a JSONL or CSV file. Each record is expected to contain at minimum a
``behavior`` field; optional fields are ``threat_category``, ``attack_type``, and ``target``.

Missing optional fields default to:
  - threat_category → "general"
  - attack_type     → "scenario_nesting"
  - target          → ""

Splits are deterministic given the same ``seed``.

Supported file formats:
  - ``.jsonl``: one JSON object per line
  - ``.csv``:  comma-separated with a header row; column names are mapped case-insensitively

Default path constant:
  DEFAULT_HARMBENCH_PATH = "../corpora/threat_categories/"
"""

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HARMBENCH_PATH = "../corpora/threat_categories/"

_DEFAULT_THREAT = "general"
_DEFAULT_ATTACK = "scenario_nesting"
_DEFAULT_TARGET = ""

# ---------------------------------------------------------------------------
# Internal record type
# ---------------------------------------------------------------------------


@dataclass
class BehaviorRecord:
    """A single HarmBench behavior record normalised for GPO consumption.

    Attributes:
        behavior: Natural-language description of the malicious goal.
        threat_category: Threat taxonomy category key (e.g. ``cybersecurity_misuse``).
        attack_type: Attack type key (e.g. ``scenario_nesting``).
        target: Optional expected prefix / target string for the model reply.
    """

    behavior: str
    threat_category: str = _DEFAULT_THREAT
    attack_type: str = _DEFAULT_ATTACK
    target: str = _DEFAULT_TARGET

    def to_dict(self) -> dict:
        """Return a plain dict representation of this record."""
        return {
            "behavior": self.behavior,
            "threat_category": self.threat_category,
            "attack_type": self.attack_type,
            "target": self.target,
        }


# ---------------------------------------------------------------------------
# AttackDataset
# ---------------------------------------------------------------------------


class AttackDataset:
    """Dataset wrapper that loads HarmBench behaviors and provides split-aware iteration.

    Args:
        filepath: Path to a ``.jsonl`` or ``.csv`` file containing behavior records.
        split_ratios: A 3-tuple ``(train, eval, test)`` of float ratios that must sum to 1.0.
        seed: Random seed used for deterministic shuffling before splitting.

    Raises:
        ValueError: If ``filepath`` does not end in ``.jsonl`` or ``.csv``, or if split ratios
            do not sum to approximately 1.0.
        FileNotFoundError: If ``filepath`` does not exist.
    """

    def __init__(
        self,
        filepath: str,
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
    ) -> None:
        self.filepath = filepath
        self.seed = seed

        train_r, eval_r, test_r = split_ratios
        if abs(train_r + eval_r + test_r - 1.0) > 1e-6:
            raise ValueError(
                f"split_ratios must sum to 1.0; got {train_r} + {eval_r} + {test_r} = "
                f"{train_r + eval_r + test_r:.6f}"
            )
        self._split_ratios = split_ratios

        self._records: list[BehaviorRecord] = self._load(filepath)
        self._train: list[BehaviorRecord] = []
        self._eval: list[BehaviorRecord] = []
        self._test: list[BehaviorRecord] = []
        self._split()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_record(raw: dict) -> BehaviorRecord:
        """Convert a raw dict (from JSON/CSV) into a :class:`BehaviorRecord`.

        Column names are compared case-insensitively.  The ``behavior`` field is mandatory;
        any record missing it is silently skipped by the caller.

        Args:
            raw: Raw key-value mapping from the file.

        Returns:
            A ``BehaviorRecord`` with defaults applied for missing optional fields.
        """
        # Normalise keys to lower-case for case-insensitive lookup
        normalised = {
            k.strip().lower(): str(v).strip() for k, v in raw.items() if v is not None
        }

        behavior = (
            normalised.get("behavior")
            or normalised.get("goal")
            or normalised.get("prompt")
            or ""
        )
        threat_category = (
            normalised.get("threat_category")
            or normalised.get("threat")
            or _DEFAULT_THREAT
        )
        attack_type = (
            normalised.get("attack_type") or normalised.get("attack") or _DEFAULT_ATTACK
        )
        target = (
            normalised.get("target") or normalised.get("target_str") or _DEFAULT_TARGET
        )

        return BehaviorRecord(
            behavior=behavior,
            threat_category=threat_category,
            attack_type=attack_type,
            target=target,
        )

    def _load(self, filepath: str) -> list[BehaviorRecord]:
        """Load records from *filepath*, auto-detecting the format by extension.

        Args:
            filepath: Path to the data file.

        Returns:
            A list of :class:`BehaviorRecord` objects (records with empty ``behavior`` skipped).

        Raises:
            FileNotFoundError: If *filepath* does not exist.
            ValueError: If the file extension is not ``.jsonl`` or ``.csv``.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"AttackDataset: file not found: {filepath}")

        ext = path.suffix.lower()
        if ext == ".jsonl":
            return self._load_jsonl(path)
        elif ext == ".csv":
            return self._load_csv(path)
        else:
            raise ValueError(
                f"AttackDataset: unsupported file extension '{ext}'. Expected '.jsonl' or '.csv'."
            )

    @staticmethod
    def _load_jsonl(path: Path) -> list[BehaviorRecord]:
        """Parse a JSONL file, one JSON object per line.

        Args:
            path: Path to the JSONL file.

        Returns:
            List of parsed :class:`BehaviorRecord` objects.
        """
        records: list[BehaviorRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[AttackDataset] WARNING: skipping malformed JSON on line {lineno}: {exc}"
                    )
                    continue
                rec = AttackDataset._normalise_record(raw)
                if rec.behavior:
                    records.append(rec)
                else:
                    print(
                        f"[AttackDataset] WARNING: skipping record on line {lineno} — empty 'behavior'."
                    )
        return records

    @staticmethod
    def _load_csv(path: Path) -> list[BehaviorRecord]:
        """Parse a CSV file with a header row.

        Args:
            path: Path to the CSV file.

        Returns:
            List of parsed :class:`BehaviorRecord` objects.
        """
        records: list[BehaviorRecord] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for rowno, row in enumerate(reader, start=2):  # row 1 is the header
                rec = AttackDataset._normalise_record(dict(row))
                if rec.behavior:
                    records.append(rec)
                else:
                    print(
                        f"[AttackDataset] WARNING: skipping row {rowno} — empty 'behavior'."
                    )
        return records

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _split(self) -> None:
        """Shuffle records with ``self.seed`` then partition into train/eval/test."""
        rng = random.Random(self.seed)
        shuffled = list(self._records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_r, eval_r, _ = self._split_ratios
        train_end = round(n * train_r)
        eval_end = train_end + round(n * eval_r)

        self._train = shuffled[:train_end]
        self._eval = shuffled[train_end:eval_end]
        self._test = shuffled[eval_end:]

    # ------------------------------------------------------------------
    # Public split accessors
    # ------------------------------------------------------------------

    def get_train(self) -> list[dict]:
        """Return the training split as a list of plain dicts.

        Returns:
            List of dicts with keys ``behavior``, ``threat_category``, ``attack_type``, ``target``.
        """
        return [r.to_dict() for r in self._train]

    def get_eval(self) -> list[dict]:
        """Return the evaluation split as a list of plain dicts.

        Returns:
            List of dicts with keys ``behavior``, ``threat_category``, ``attack_type``, ``target``.
        """
        return [r.to_dict() for r in self._eval]

    def get_test(self) -> list[dict]:
        """Return the test split as a list of plain dicts.

        Returns:
            List of dicts with keys ``behavior``, ``threat_category``, ``attack_type``, ``target``.
        """
        return [r.to_dict() for r in self._test]

    def get_format_data(self, n: int = 5) -> list[dict]:
        """Return up to *n* records from the training split to use as formatting examples.

        Records are taken from the front of the (already shuffled) training split, so the
        selection is deterministic for a given ``seed``.

        Args:
            n: Maximum number of records to return.

        Returns:
            List of dicts (possibly fewer than *n* if the training split is small).
        """
        return [r.to_dict() for r in self._train[:n]]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of loaded records (across all splits)."""
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"AttackDataset(filepath={self.filepath!r}, total={len(self)}, "
            f"train={len(self._train)}, eval={len(self._eval)}, test={len(self._test)})"
        )

    @classmethod
    def from_records(
        cls,
        records: list[dict],
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
    ) -> "AttackDataset":
        """Construct an :class:`AttackDataset` directly from an in-memory list of dicts.

        This is useful when behaviors are already loaded (e.g. from the taxonomy module)
        and a file path is not available.

        Args:
            records: List of dicts, each with at least a ``"behavior"`` key.
            split_ratios: Train/eval/test split ratios (must sum to 1.0).
            seed: Random seed for shuffling.

        Returns:
            A fully initialised :class:`AttackDataset`.
        """
        # Create a minimal instance and populate directly without file I/O
        instance = object.__new__(cls)
        instance.filepath = "<in-memory>"
        instance.seed = seed

        train_r, eval_r, test_r = split_ratios
        if abs(train_r + eval_r + test_r - 1.0) > 1e-6:
            raise ValueError(
                f"split_ratios must sum to 1.0; got sum = {train_r + eval_r + test_r:.6f}"
            )
        instance._split_ratios = split_ratios
        instance._records = [
            cls._normalise_record(r) for r in records if r.get("behavior")
        ]
        instance._train = []
        instance._eval = []
        instance._test = []
        instance._split()
        return instance
