from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def hidden_subprocess_kwargs() -> dict[str, Any]:
    if not hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Required tool not found on PATH: {name}")
    return found


def ffprobe(path: Path) -> dict[str, Any]:
    require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **hidden_subprocess_kwargs(),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffprobe failed: {path}")
    data = json.loads(proc.stdout)
    audio_stream = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    if not audio_stream:
        raise RuntimeError(f"No audio stream: {path}")
    duration = audio_stream.get("duration") or data.get("format", {}).get("duration")
    return {
        "duration_sec": float(duration) if duration else None,
        "sample_rate": int(audio_stream["sample_rate"]) if audio_stream.get("sample_rate") else None,
        "channels": int(audio_stream["channels"]) if audio_stream.get("channels") else None,
    }


def decode_to_wav(src: Path, dst: Path, sample_rate: int = 24000) -> None:
    require_tool("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **hidden_subprocess_kwargs(),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg decode failed: {src}")


def read_audio(path: Path, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required to read decoded wav files") from exc

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        try:
            librosa = __import__("librosa")

            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as exc:
            raise RuntimeError(f"Cannot resample {path} from {sr} to {target_sr}") from exc
    return np.asarray(wav, dtype=np.float32), sr


def write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="PCM_16")


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def dbfs(value: float) -> float:
    if value <= 1e-12:
        return -120.0
    return float(20.0 * math.log10(value))
