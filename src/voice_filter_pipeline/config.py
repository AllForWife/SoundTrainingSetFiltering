from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}


@dataclass(frozen=True)
class Paths:
    root: Path
    clips: Path
    work: Path
    out: Path
    manifest: Path
    wav_cache: Path

    @classmethod
    def from_root(cls, root: Path, clips_dir: Path | None = None) -> "Paths":
        root = root.resolve()
        work = root / "work"
        return cls(
            root=root,
            clips=(clips_dir.resolve() if clips_dir else root / "clips"),
            work=work,
            out=root / "out",
            manifest=work / "manifests" / "master.jsonl",
            wav_cache=work / "cache" / "wav_24k_mono",
        )


@dataclass(frozen=True)
class Thresholds:
    min_duration_sec: float = 0.8
    min_speech_ratio: float = 0.35
    max_silence_ratio: float = 0.65
    max_clipping_ratio: float = 0.002
    female_quality_pass: float = 0.78
    female_quality_review: float = 0.70
    post_uvr_pass: float = 0.78
    post_uvr_review: float = 0.70
    female_f0_hz: float = 165.0
    male_f0_hz: float = 155.0
