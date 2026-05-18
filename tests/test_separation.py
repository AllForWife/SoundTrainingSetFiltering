from __future__ import annotations

from pathlib import Path

from voice_filter_pipeline.separation import detect_backend


def test_detect_backend_uses_configured_uvr_python(monkeypatch) -> None:
    python_path = Path("work") / "fake_uvr" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("VOICE_FILTER_UVR_PYTHON", str(python_path))
    assert detect_backend("audio-separator") == "audio-separator"
