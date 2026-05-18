from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from voice_filter_pipeline.pipeline import Pipeline, resolve_worker_count


def test_default_checkpoint_interval_is_500() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    try:
        pipeline = Pipeline(root=tmp_path)
        assert pipeline.checkpoint_interval == 500
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_worker_count_is_bounded_and_overridable() -> None:
    assert resolve_worker_count(3) == 3
    assert resolve_worker_count(0) == 1
    assert 1 <= resolve_worker_count(None) <= 8


def test_checkpoint_interval_uses_every_n_records() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    try:
        pipeline = Pipeline(root=tmp_path, checkpoint_interval=2)
        assert not pipeline._should_checkpoint(1)
        assert pipeline._should_checkpoint(2)
        assert not pipeline._should_checkpoint(3)
        assert pipeline._should_checkpoint(4)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_existing_female_raw_output_marks_score_done() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    try:
        pipeline = Pipeline(root=tmp_path)
        src = tmp_path / "clips" / "a.mp3"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"audio")
        row = {"source_path": str(src), "sha1": "abc123"}
        raw = tmp_path / "out" / "female_raw_pass" / "abc123.wav"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"wav")
        assert pipeline._apply_existing_score_output(row)
        assert row["final_status"] == "female_raw_pass"
        assert Path(row["female_raw_path"]) == raw.resolve()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_existing_manual_uvr_output_marks_file_skipped() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    try:
        pipeline = Pipeline(root=tmp_path)
        src = tmp_path / "input" / "a.wav"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"audio")
        row = {"source_path": str(src), "sha1": "def456"}
        clean = tmp_path / "out" / "uvr_cleaned_selected" / "def456.wav"
        clean.parent.mkdir(parents=True)
        clean.write_bytes(b"wav")
        assert pipeline._apply_existing_manual_uvr_output(row)
        assert row["final_status"] == "manual_uvr_cleaned"
        assert Path(row["final_output_path"]) == clean.resolve()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_progress_callback_gets_eta_payload() -> None:
    tmp_path = Path("work") / "test_tmp" / uuid.uuid4().hex
    events: list[dict] = []
    try:
        pipeline = Pipeline(root=tmp_path, progress_callback=events.append, progress_interval=2)
        started_at = 0.0
        pipeline._emit_progress("scan", 2, 10, started_at, Path("a.mp3"), force=True)
        assert events
        event = events[-1]
        assert event["stage"] == "scan"
        assert event["processed"] == 2
        assert event["total"] == 10
        assert event["eta_sec"] is not None
        assert event["current_path"] == "a.mp3"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
