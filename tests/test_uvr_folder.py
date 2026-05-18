from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from voice_filter_pipeline.pipeline import Pipeline


def test_uvr_folder_selects_audio_files(monkeypatch) -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    folder = tmp_path / "audio"
    folder.mkdir(parents=True)
    first = folder / "a.wav"
    second = folder / "b.txt"
    try:
        first.write_bytes(b"wav")
        second.write_text("ignore", encoding="utf-8")
        pipeline = Pipeline(root=tmp_path)
        seen: list[Path] = []

        def fake_uvr(files, backend="auto"):
            seen.extend(files)
            return {"processed": len(files), "accepted": 0, "failed": 0}

        monkeypatch.setattr(pipeline, "uvr_selected_files", fake_uvr)
        result = pipeline.uvr_folder(folder)
        assert result["processed"] == 1
        assert seen == [first.resolve()]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
