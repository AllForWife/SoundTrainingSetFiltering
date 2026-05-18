from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import uuid


def load_gui_module():
    spec = importlib.util.spec_from_file_location("gui_voice_filter", Path("gui_voice_filter.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_limit() -> None:
    module = load_gui_module()
    assert module.parse_limit("") is None
    assert module.parse_limit("25") == 25
    assert module.resolve_limit("25", False) == 25
    assert module.resolve_limit("25", True) is None
    assert module.parse_workers("") is None
    assert module.parse_workers("auto") is None
    assert module.parse_workers("4") == 4
    assert module.parse_uvr_batch_size("") is None
    assert module.parse_uvr_batch_size("auto") is None
    assert module.parse_uvr_batch_size("8") == 8


def test_format_progress_includes_eta_and_current_file() -> None:
    module = load_gui_module()
    text = module.format_progress(
        {
            "stage": "scan",
            "processed": 100,
            "total": 1000,
            "percent": 10.0,
            "rate_per_min": 300.0,
            "elapsed_sec": 20.0,
            "eta_sec": 180.0,
            "current_path": r"D:\clips\a.mp3",
        }
    )
    assert "掃描音檔" in text
    assert "100/1000" in text
    assert "預計剩餘" in text
    assert "a.mp3" in text


def test_gui_smoke_main() -> None:
    module = load_gui_module()
    assert module.main(["--smoke"]) == 0


def test_app_root_uses_project_parent_when_frozen_in_dist(monkeypatch) -> None:
    module = load_gui_module()
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    project = tmp_path / "project"
    dist = project / "dist"
    try:
        dist.mkdir(parents=True)
        (project / ".venv-uvr").mkdir()
        monkeypatch.setattr(module.sys, "frozen", True, raising=False)
        monkeypatch.setattr(module.sys, "executable", str(dist / "VoiceFilterGUI.exe"))
        assert module.app_root() == project.resolve()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
