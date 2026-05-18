from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import os
from pathlib import Path
import time
from typing import Any, Callable

from .audio import decode_to_wav, ffprobe, read_audio, write_wav
from .config import Paths, Thresholds
from .features import classify_gender, extract_features, quality_score
from .fs import find_audio_files, output_name, safe_copy, select_files, sha1_file
from .manifest import default_row, load_manifest, write_manifest
from .separation import separate_vocal


class Pipeline:
    def __init__(
        self,
        root: Path,
        thresholds: Thresholds | None = None,
        clips_dir: Path | None = None,
        input_files: list[Path] | None = None,
        checkpoint_interval: int = 500,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval: int = 100,
        max_workers: int | None = None,
    ) -> None:
        self.paths = Paths.from_root(root, clips_dir)
        self.thresholds = thresholds or Thresholds()
        self.input_files = [path.resolve() for path in input_files] if input_files else None
        self.checkpoint_interval = max(1, checkpoint_interval)
        self.progress_callback = progress_callback
        self.progress_interval = max(1, progress_interval)
        self.max_workers = resolve_worker_count(max_workers)

    def scan(self, limit: int | None = None, seed: int = 7, randomize: bool = False) -> dict[str, Any]:
        rows = load_manifest(self.paths.manifest)
        source_files = self.input_files if self.input_files is not None else find_audio_files(self.paths.clips)
        files = select_files(source_files, limit, seed, randomize)
        started_at = time.monotonic()
        self._emit_progress("scan", 0, len(files), started_at, force=True)
        processed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._scan_one, src, rows.get(str(src.resolve()))) for src in files]
            for future in as_completed(futures):
                key, row, current_path = future.result()
                processed += 1
                rows[key] = row
                if self._should_checkpoint(processed):
                    write_manifest(self.paths.manifest, rows)
                    self._emit_progress("scan", processed, len(files), started_at, current_path, checkpoint=True)
                else:
                    self._emit_progress("scan", processed, len(files), started_at, current_path)
        write_manifest(self.paths.manifest, rows)
        self._emit_progress("scan", len(files), len(files), started_at, done=True, force=True)
        return {
            "processed": len(files),
            "checkpoint_interval": self.checkpoint_interval,
            "workers": self.max_workers,
            "manifest": str(self.paths.manifest),
        }

    def classify(self, limit: int | None = None, seed: int = 7, randomize: bool = False) -> dict[str, Any]:
        rows = load_manifest(self.paths.manifest)
        selected = self._select_rows(rows, limit, seed, randomize, lambda row: row.get("decode_ok") and row.get("gender") is None)
        started_at = time.monotonic()
        self._emit_progress("classify", 0, len(selected), started_at, force=True)
        processed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._classify_one, dict(row)) for row in selected]
            for future in as_completed(futures):
                row, current_path = future.result()
                processed += 1
                rows[str(Path(str(row["source_path"])).resolve())] = row
                if self._should_checkpoint(processed):
                    write_manifest(self.paths.manifest, rows)
                    self._emit_progress("classify", processed, len(selected), started_at, current_path, checkpoint=True)
                else:
                    self._emit_progress("classify", processed, len(selected), started_at, current_path)
        write_manifest(self.paths.manifest, rows)
        self._emit_progress("classify", len(selected), len(selected), started_at, done=True, force=True)
        return {"processed": len(selected), "checkpoint_interval": self.checkpoint_interval, "workers": self.max_workers}

    def score(self, limit: int | None = None, seed: int = 7, randomize: bool = False) -> dict[str, Any]:
        rows = load_manifest(self.paths.manifest)
        selected = self._select_rows(
            rows,
            limit,
            seed,
            randomize,
            lambda row: row.get("final_status") in {"female_candidate", "quality_borderline"},
        )
        started_at = time.monotonic()
        self._emit_progress("score", 0, len(selected), started_at, force=True)
        processed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._score_one, dict(row)) for row in selected]
            for future in as_completed(futures):
                row, current_path = future.result()
                processed += 1
                rows[str(Path(str(row["source_path"])).resolve())] = row
                if self._should_checkpoint(processed):
                    write_manifest(self.paths.manifest, rows)
                    self._emit_progress("score", processed, len(selected), started_at, current_path, checkpoint=True)
                else:
                    self._emit_progress("score", processed, len(selected), started_at, current_path)
        write_manifest(self.paths.manifest, rows)
        self._emit_progress("score", len(selected), len(selected), started_at, done=True, force=True)
        return {"processed": len(selected), "checkpoint_interval": self.checkpoint_interval, "workers": self.max_workers}

    def separate(
        self,
        limit: int | None = None,
        seed: int = 7,
        randomize: bool = False,
        backend: str = "auto",
    ) -> dict[str, Any]:
        rows = load_manifest(self.paths.manifest)
        selected = self._select_rows(
            rows,
            limit,
            seed,
            randomize,
            lambda row: row.get("final_status") in {"female_raw_pass", "uvr_failed"},
        )
        started_at = time.monotonic()
        self._emit_progress("separate", 0, len(selected), started_at, force=True)
        for processed, row in enumerate(selected, start=1):
            src = Path(row.get("female_raw_path") or self._ensure_wav(row))
            dst = self.paths.work / "uvr" / output_name(row, ".wav")
            if dst.exists():
                row["uvr_output_path"] = str(dst)
                row["final_status"] = "uvr_done"
                row["reject_reason"] = None
                if self._should_checkpoint(processed):
                    write_manifest(self.paths.manifest, rows)
                    self._emit_progress("separate", processed, len(selected), started_at, src, checkpoint=True)
                else:
                    self._emit_progress("separate", processed, len(selected), started_at, src)
                continue
            try:
                ok, selected_backend, error = separate_vocal(src, dst, backend)
                row["uvr_backend"] = selected_backend
                if ok:
                    row["uvr_output_path"] = str(dst)
                    row["final_status"] = "uvr_done"
                    row["reject_reason"] = None
                else:
                    row["uvr_output_path"] = None
                    row["final_status"] = "uvr_failed"
                    row["reject_reason"] = error or "uvr_failed"
            except Exception as exc:
                row["final_status"] = "uvr_failed"
                row["reject_reason"] = f"uvr_failed:{exc}"
            if self._should_checkpoint(processed):
                write_manifest(self.paths.manifest, rows)
                self._emit_progress("separate", processed, len(selected), started_at, src, checkpoint=True)
            else:
                self._emit_progress("separate", processed, len(selected), started_at, src)
        write_manifest(self.paths.manifest, rows)
        self._emit_progress("separate", len(selected), len(selected), started_at, done=True, force=True)
        return {"processed": len(selected), "checkpoint_interval": self.checkpoint_interval, "workers": 1}

    def verify(self, limit: int | None = None, seed: int = 7, randomize: bool = False) -> dict[str, Any]:
        rows = load_manifest(self.paths.manifest)
        selected = self._select_rows(rows, limit, seed, randomize, lambda row: row.get("final_status") == "uvr_done")
        started_at = time.monotonic()
        self._emit_progress("verify", 0, len(selected), started_at, force=True)
        processed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._verify_one, dict(row)) for row in selected]
            for future in as_completed(futures):
                row, current_path = future.result()
                processed += 1
                rows[str(Path(str(row["source_path"])).resolve())] = row
                if self._should_checkpoint(processed):
                    write_manifest(self.paths.manifest, rows)
                    self._emit_progress("verify", processed, len(selected), started_at, current_path, checkpoint=True)
                else:
                    self._emit_progress("verify", processed, len(selected), started_at, current_path)
        write_manifest(self.paths.manifest, rows)
        self._emit_progress("verify", len(selected), len(selected), started_at, done=True, force=True)
        report_path = self.write_report(rows)
        return {
            "processed": len(selected),
            "checkpoint_interval": self.checkpoint_interval,
            "workers": self.max_workers,
            "report": str(report_path),
        }

    def run_all(
        self,
        limit: int | None = None,
        seed: int = 7,
        randomize: bool = False,
        backend: str = "auto",
    ) -> dict[str, Any]:
        return self.filter_only(limit, seed, randomize)

    def filter_only(
        self,
        limit: int | None = None,
        seed: int = 7,
        randomize: bool = False,
    ) -> dict[str, Any]:
        return {
            "scan": self.scan(limit, seed, randomize),
            "classify": self.classify(limit, seed, randomize),
            "score": self.score(limit, seed, randomize),
        }

    def uvr_selected_files(
        self,
        files: list[Path],
        backend: str = "auto",
    ) -> dict[str, Any]:
        rows = load_manifest(self.paths.manifest)
        processed = 0
        accepted = 0
        failed = 0
        skipped = 0
        started_at = time.monotonic()
        self._emit_progress("manual_uvr", 0, len(files), started_at, force=True)
        for src in files:
            src = src.resolve()
            row = rows.get(str(src), default_row(src))
            row["sha1"] = row.get("sha1") or sha1_file(src)
            if self._apply_existing_manual_uvr_output(row):
                skipped += 1
                processed += 1
                if self._should_checkpoint(processed):
                    write_manifest(self.paths.manifest, rows)
                    self._emit_progress("manual_uvr", processed, len(files), started_at, src, checkpoint=True)
                else:
                    self._emit_progress("manual_uvr", processed, len(files), started_at, src)
                continue
            row["decode_ok"] = True
            row["manual_uvr_source_path"] = str(src)
            dst = self.paths.work / "uvr_manual" / output_name(row, ".wav")
            try:
                ok, selected_backend, error = separate_vocal(src, dst, backend)
                row["uvr_backend"] = selected_backend
                if not ok:
                    failed += 1
                    row["uvr_output_path"] = None
                    row["final_status"] = "manual_uvr_failed"
                    row["reject_reason"] = error or "uvr_failed"
                    rows[str(src)] = row
                    continue
                row["uvr_output_path"] = str(dst)
                wav, sr = read_audio(dst)
                features = extract_features(wav, sr)
                score, reasons = quality_score(features, self.thresholds)
                row["post_uvr_score"] = score
                row["post_uvr_reasons"] = reasons
                row["post_uvr_feature_summary"] = _feature_summary(features)
                clean_path = self.paths.out / "uvr_cleaned_selected" / output_name(row, ".wav")
                write_wav(clean_path, wav, sr)
                row["final_output_path"] = str(clean_path)
                row["final_status"] = "manual_uvr_cleaned"
                row["reject_reason"] = None if score >= self.thresholds.post_uvr_review else "manual_uvr_low_score"
                accepted += 1
            except Exception as exc:
                failed += 1
                row["final_status"] = "manual_uvr_failed"
                row["reject_reason"] = f"uvr_failed:{exc}"
            finally:
                processed += 1
                rows[str(src)] = row
                if self._should_checkpoint(processed):
                    write_manifest(self.paths.manifest, rows)
                    self._emit_progress("manual_uvr", processed, len(files), started_at, src, checkpoint=True)
                else:
                    self._emit_progress("manual_uvr", processed, len(files), started_at, src)
        write_manifest(self.paths.manifest, rows)
        self._emit_progress("manual_uvr", len(files), len(files), started_at, done=True, force=True)
        report_path = self.write_report(rows)
        return {
            "processed": processed,
            "accepted": accepted,
            "failed": failed,
            "skipped": skipped,
            "checkpoint_interval": self.checkpoint_interval,
            "report": str(report_path),
        }

    def uvr_folder(
        self,
        folder: Path,
        limit: int | None = None,
        seed: int = 7,
        randomize: bool = False,
        backend: str = "auto",
    ) -> dict[str, Any]:
        files = select_files(find_audio_files(folder), limit, seed, randomize)
        result = self.uvr_selected_files(files, backend=backend)
        result["folder"] = str(folder.resolve())
        return result

    def write_report(self, rows: dict[str, dict[str, Any]]) -> Path:
        self.paths.out.mkdir(parents=True, exist_ok=True)
        report = self.paths.out / "report.md"
        statuses = Counter(str(row.get("final_status")) for row in rows.values())
        rejects = Counter(str(row.get("reject_reason")) for row in rows.values() if row.get("reject_reason"))
        lines = [
            "# Voice Filter Report",
            "",
            f"- Total manifest rows: {len(rows)}",
            f"- Clean trainable: {statuses.get('clean_trainable', 0)}",
            f"- Manual UVR cleaned: {statuses.get('manual_uvr_cleaned', 0)}",
            f"- Female raw pass: {statuses.get('female_raw_pass', 0)}",
            f"- UVR done pending verify: {statuses.get('uvr_done', 0)}",
            f"- UVR failed: {statuses.get('uvr_failed', 0)}",
            "",
            "## Status counts",
            "",
        ]
        lines += [f"- {key}: {value}" for key, value in statuses.most_common()]
        lines += ["", "## Reject reasons", ""]
        lines += [f"- {key}: {value}" for key, value in rejects.most_common(30)]
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def _scan_one(self, src: Path, existing_row: dict[str, Any] | None) -> tuple[str, dict[str, Any], Path]:
        src = src.resolve()
        key = str(src)
        row = dict(existing_row) if existing_row else default_row(src)
        if row.get("decode_ok") and row.get("cached_wav_path") and Path(str(row["cached_wav_path"])).exists():
            return key, row, src
        try:
            row.update(ffprobe(src))
            row["sha1"] = row.get("sha1") or sha1_file(src)
            wav_path = self._wav_cache_path(row)
            if not wav_path.exists():
                decode_to_wav(src, wav_path)
            row["decode_ok"] = True
            row["cached_wav_path"] = str(wav_path)
            if row.get("final_status") is None:
                row["final_status"] = "scanned"
                row["reject_reason"] = None
        except Exception as exc:
            row["decode_ok"] = False
            row["final_status"] = "rejected"
            row["reject_reason"] = f"decode_failed:{exc}"
        return key, row, src

    def _classify_one(self, row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        current_path = Path(str(row["source_path"]))
        if self._apply_existing_classify_output(row):
            return row, current_path
        wav_path = self._ensure_wav(row)
        wav, sr = read_audio(wav_path)
        features = extract_features(wav, sr)
        gender, confidence, reason = classify_gender(features, self.thresholds)
        row["speech_ratio"] = round(features.speech_ratio, 4)
        row["median_f0_hz"] = round(features.median_f0_hz, 2) if features.median_f0_hz else None
        row["gender"] = gender
        row["gender_confidence"] = round(confidence, 4)
        row["feature_summary"] = _feature_summary(features)
        if reason == "no_voice":
            row["final_status"] = "rejected"
            row["reject_reason"] = "no_voice"
            self._copy_source(row, self.paths.out / "rejected" / "no_voice")
        elif gender == "male" and confidence >= 0.70:
            row["final_status"] = "rejected"
            row["reject_reason"] = "male"
            self._copy_source(row, self.paths.out / "rejected" / "male")
        elif gender != "female" or confidence < 0.65:
            row["final_status"] = "review"
            row["reject_reason"] = reason or "gender_uncertain"
            self._copy_source(row, self.paths.out / "review" / "gender_uncertain")
        else:
            row["final_status"] = "female_candidate"
            row["reject_reason"] = None
        return row, current_path

    def _score_one(self, row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        current_path = Path(str(row["source_path"]))
        if self._apply_existing_score_output(row):
            return row, current_path
        wav_path = self._ensure_wav(row)
        wav, sr = read_audio(wav_path)
        features = extract_features(wav, sr)
        score, reasons = quality_score(features, self.thresholds)
        row["quality_score"] = score
        row["quality_reasons"] = reasons
        row["feature_summary"] = _feature_summary(features)
        if score >= self.thresholds.female_quality_pass and not _hard_quality_fail(reasons):
            row["final_status"] = "female_raw_pass"
            row["reject_reason"] = None
            raw_path = self.paths.out / "female_raw_pass" / output_name(row, ".wav")
            write_wav(raw_path, wav, sr)
            row["female_raw_path"] = str(raw_path)
        elif score >= self.thresholds.female_quality_review:
            row["final_status"] = "review"
            row["reject_reason"] = "quality_borderline"
            self._copy_source(row, self.paths.out / "review" / "quality_borderline")
        else:
            row["final_status"] = "rejected"
            row["reject_reason"] = "low_quality:" + ",".join(reasons)
            self._copy_source(row, self.paths.out / "rejected" / "low_quality")
        return row, current_path

    def _verify_one(self, row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        uvr_path = Path(str(row["uvr_output_path"]))
        if self._apply_existing_verify_output(row):
            return row, uvr_path
        wav, sr = read_audio(uvr_path)
        features = extract_features(wav, sr)
        score, reasons = quality_score(features, self.thresholds)
        row["post_uvr_score"] = score
        row["post_uvr_reasons"] = reasons
        row["post_uvr_feature_summary"] = _feature_summary(features)
        if score >= self.thresholds.post_uvr_pass and not _hard_quality_fail(reasons):
            dst = self.paths.out / "female_clean_trainable" / output_name(row, ".wav")
            write_wav(dst, wav, sr)
            row["final_output_path"] = str(dst)
            row["final_status"] = "clean_trainable"
            row["reject_reason"] = None
        elif score >= self.thresholds.post_uvr_review:
            row["final_status"] = "review"
            row["reject_reason"] = "uvr_borderline"
            self._copy_audio(uvr_path, self.paths.out / "review" / "uvr_borderline" / output_name(row, ".wav"))
        else:
            row["final_status"] = "rejected"
            row["reject_reason"] = "uvr_artifact:" + ",".join(reasons)
            self._copy_audio(uvr_path, self.paths.out / "rejected" / "uvr_artifact" / output_name(row, ".wav"))
        return row, uvr_path

    def _wav_cache_path(self, row: dict[str, Any]) -> Path:
        stem = row.get("sha1") or Path(str(row["source_path"])).stem
        return self.paths.wav_cache / f"{stem}.wav"

    def _ensure_wav(self, row: dict[str, Any]) -> Path:
        wav_path = Path(str(row.get("cached_wav_path") or self._wav_cache_path(row)))
        if not wav_path.exists():
            decode_to_wav(Path(str(row["source_path"])), wav_path)
            row["cached_wav_path"] = str(wav_path)
        return wav_path

    def _select_rows(self, rows: dict[str, dict[str, Any]], limit, seed, randomize, predicate):
        values = [row for row in rows.values() if predicate(row)]
        paths = [Path(str(row["source_path"])) for row in values]
        selected_paths = {str(path) for path in select_files(paths, limit, seed, randomize)}
        return [row for row in values if str(Path(str(row["source_path"]))) in selected_paths]

    def _copy_source(self, row: dict[str, Any], dst_dir: Path) -> None:
        src = Path(str(row["source_path"]))
        safe_copy(src, dst_dir / src.name)

    def _copy_audio(self, src: Path, dst: Path) -> None:
        safe_copy(src, dst)

    def _should_checkpoint(self, processed: int) -> bool:
        return processed > 0 and processed % self.checkpoint_interval == 0

    def _apply_existing_classify_output(self, row: dict[str, Any]) -> bool:
        if self._source_copy_exists(row, "rejected", "no_voice"):
            row["final_status"] = "rejected"
            row["reject_reason"] = "no_voice"
            row["gender"] = "unknown"
            return True
        if self._source_copy_exists(row, "rejected", "male"):
            row["final_status"] = "rejected"
            row["reject_reason"] = "male"
            row["gender"] = "male"
            return True
        if self._source_copy_exists(row, "review", "gender_uncertain"):
            row["final_status"] = "review"
            row["reject_reason"] = "gender_uncertain"
            row["gender"] = "unknown"
            return True
        return False

    def _apply_existing_score_output(self, row: dict[str, Any]) -> bool:
        raw_path = self.paths.out / "female_raw_pass" / output_name(row, ".wav")
        if raw_path.exists():
            row["final_status"] = "female_raw_pass"
            row["reject_reason"] = None
            row["female_raw_path"] = str(raw_path)
            return True
        if self._source_copy_exists(row, "review", "quality_borderline"):
            row["final_status"] = "review"
            row["reject_reason"] = "quality_borderline"
            return True
        if self._source_copy_exists(row, "rejected", "low_quality"):
            row["final_status"] = "rejected"
            row["reject_reason"] = row.get("reject_reason") or "low_quality:existing_output"
            return True
        return False

    def _apply_existing_verify_output(self, row: dict[str, Any]) -> bool:
        clean_path = self.paths.out / "female_clean_trainable" / output_name(row, ".wav")
        if clean_path.exists():
            row["final_output_path"] = str(clean_path)
            row["final_status"] = "clean_trainable"
            row["reject_reason"] = None
            return True
        borderline_path = self.paths.out / "review" / "uvr_borderline" / output_name(row, ".wav")
        if borderline_path.exists():
            row["final_status"] = "review"
            row["reject_reason"] = "uvr_borderline"
            return True
        artifact_path = self.paths.out / "rejected" / "uvr_artifact" / output_name(row, ".wav")
        if artifact_path.exists():
            row["final_status"] = "rejected"
            row["reject_reason"] = row.get("reject_reason") or "uvr_artifact:existing_output"
            return True
        return False

    def _apply_existing_manual_uvr_output(self, row: dict[str, Any]) -> bool:
        clean_path = self.paths.out / "uvr_cleaned_selected" / output_name(row, ".wav")
        if not clean_path.exists():
            return False
        row["decode_ok"] = True
        row["manual_uvr_source_path"] = str(row["source_path"])
        row["final_output_path"] = str(clean_path)
        row["final_status"] = "manual_uvr_cleaned"
        row["reject_reason"] = None
        return True

    def _source_copy_exists(self, row: dict[str, Any], *parts: str) -> bool:
        return (self.paths.out.joinpath(*parts) / Path(str(row["source_path"])).name).exists()

    def _should_emit_progress(self, processed: int, total: int, force: bool) -> bool:
        return force or processed == 0 or processed == total or processed % self.progress_interval == 0

    def _emit_progress(
        self,
        stage: str,
        processed: int,
        total: int,
        started_at: float,
        current_path: Path | None = None,
        checkpoint: bool = False,
        done: bool = False,
        force: bool = False,
    ) -> None:
        if self.progress_callback is None:
            return
        if not checkpoint and not done and not self._should_emit_progress(processed, total, force):
            return
        elapsed_sec = max(0.0, time.monotonic() - started_at)
        rate_per_sec = processed / elapsed_sec if processed > 0 and elapsed_sec > 0 else 0.0
        remaining = max(total - processed, 0)
        eta_sec = remaining / rate_per_sec if rate_per_sec > 0 else None
        self.progress_callback(
            {
                "stage": stage,
                "processed": processed,
                "total": total,
                "percent": (processed / total * 100.0) if total else 100.0,
                "elapsed_sec": elapsed_sec,
                "eta_sec": eta_sec,
                "rate_per_min": rate_per_sec * 60.0,
                "current_path": str(current_path) if current_path else None,
                "checkpoint": checkpoint,
                "done": done,
            }
        )


def _feature_summary(features) -> dict[str, Any]:
    return {
        "duration_sec": round(features.duration_sec, 4),
        "silence_ratio": round(features.silence_ratio, 4),
        "clipping_ratio": round(features.clipping_ratio, 6),
        "rms_dbfs": round(features.rms_dbfs, 3),
        "noise_dbfs": round(features.noise_dbfs, 3),
        "snr_db": round(features.snr_db, 3),
        "bandwidth_score": round(features.bandwidth_score, 4),
    }


def _hard_quality_fail(reasons: list[str]) -> bool:
    hard = {"too_short", "low_speech_ratio", "clipping"}
    return any(reason in hard for reason in reasons)


def resolve_worker_count(max_workers: int | None = None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count))
