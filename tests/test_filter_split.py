from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from voice_filter_pipeline.pipeline import Pipeline


def test_run_all_is_filter_only(monkeypatch) -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(root=tmp_path)
    try:
        calls: list[str] = []
        monkeypatch.setattr(pipeline, "scan", lambda *args, **kwargs: calls.append("scan") or {})
        monkeypatch.setattr(pipeline, "classify", lambda *args, **kwargs: calls.append("classify") or {})
        monkeypatch.setattr(pipeline, "score", lambda *args, **kwargs: calls.append("score") or {})
        monkeypatch.setattr(pipeline, "separate", lambda *args, **kwargs: calls.append("separate") or {})
        monkeypatch.setattr(pipeline, "verify", lambda *args, **kwargs: calls.append("verify") or {})
        pipeline.run_all()
        assert calls == ["scan", "classify", "score"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
