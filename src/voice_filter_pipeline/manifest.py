from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_path = row.get("source_path")
            if source_path:
                rows[str(source_path)] = row
    return rows


def write_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda row: str(row.get("source_path", "")))
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temp_path.replace(path)


def default_row(source_path: Path) -> dict[str, Any]:
    return {
        "source_path": str(source_path.resolve()),
        "sha1": None,
        "duration_sec": None,
        "sample_rate": None,
        "channels": None,
        "decode_ok": False,
        "speech_ratio": None,
        "gender": None,
        "gender_confidence": None,
        "quality_score": None,
        "quality_reasons": [],
        "uvr_backend": None,
        "uvr_output_path": None,
        "post_uvr_score": None,
        "final_status": None,
        "reject_reason": None,
    }
