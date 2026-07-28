"""Validated registry of manuscript results used only as frozen context."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .schema import CellKey


@dataclass(frozen=True)
class FrozenResult:
    table: int
    pdf_page: int
    row: str
    column: str
    value: float
    provenance: str = "frozen_pdf"
    cell_key: CellKey | None = None


@dataclass(frozen=True)
class FrozenRegistry:
    results: tuple[FrozenResult, ...]
    pdf_sha256: str

    @classmethod
    def load(cls, path: str | Path, pdf_path: str | Path) -> "FrozenRegistry":
        payload: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        expected = payload.get("pdf", {}).get("sha256")
        actual = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("frozen PDF hash does not match registry")
        seen: set[tuple[int, int, str, str]] = set()
        results = []
        for item in payload.get("results", []):
            locator = (item["table"], item["pdf_page"], item["row"], item["column"])
            if locator in seen:
                raise ValueError(f"duplicate frozen result locator: {locator!r}")
            seen.add(locator)
            results.append(FrozenResult(table=item["table"], pdf_page=item["pdf_page"], row=item["row"], column=item["column"], value=float(item["value"])))
        return cls(tuple(results), actual)

    def find_exact(self, requested: CellKey) -> FrozenResult | None:
        return next((row for row in self.results if row.cell_key == requested), None)

    def context_rows(self, *, table: int, row: str, column: str) -> list[FrozenResult]:
        return [item for item in self.results if (item.table, item.row, item.column) == (table, row, column)]
