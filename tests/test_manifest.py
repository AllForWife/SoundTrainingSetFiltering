from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from voice_filter_pipeline.manifest import default_row, load_manifest, write_manifest


def test_manifest_roundtrip() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "a.wav"
    row = default_row(src)
    row["sha1"] = "abc"
    manifest = tmp_path / "work" / "master.jsonl"
    try:
        write_manifest(manifest, {str(src): row})
        loaded = load_manifest(manifest)
        assert loaded[str(src.resolve())]["sha1"] == "abc"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
