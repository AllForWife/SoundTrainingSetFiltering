from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from voice_filter_pipeline.separation import _collect_batch_outputs, detect_backend


def test_detect_backend_uses_configured_uvr_python(monkeypatch) -> None:
    python_path = Path("work") / "fake_uvr" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("VOICE_FILTER_UVR_PYTHON", str(python_path))
    assert detect_backend("audio-separator") == "audio-separator"


def test_collect_batch_outputs_maps_vocals_to_sources() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    try:
        first = tmp_path / "input" / "a.wav"
        second = tmp_path / "input" / "b.wav"
        first.parent.mkdir(parents=True)
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        out = tmp_path / "out"
        out.mkdir()
        a_vocal = out / "a_(Vocals).wav"
        b_vocal = out / "b_(Vocals).wav"
        a_vocal.write_bytes(b"a")
        b_vocal.write_bytes(b"b")
        mapped = _collect_batch_outputs([first, second], out)
        assert mapped[first.resolve()] == a_vocal
        assert mapped[second.resolve()] == b_vocal
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
