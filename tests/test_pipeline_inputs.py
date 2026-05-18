from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from voice_filter_pipeline.pipeline import Pipeline


def test_pipeline_accepts_selected_input_files() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    audio = tmp_path / "x.mp3"
    try:
        audio.write_bytes(b"not-real-audio")
        pipeline = Pipeline(root=tmp_path, input_files=[audio])
        assert pipeline.input_files == [audio.resolve()]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
