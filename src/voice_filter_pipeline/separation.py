from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path

from .audio import decode_to_wav


def detect_backend(preferred: str = "auto") -> str | None:
    if preferred != "auto":
        if _backend_command(preferred):
            return preferred
        return None
    for name in ("audio-separator", "demucs"):
        if _backend_command(name):
            return name
    return None


def _backend_command(name: str) -> list[str] | None:
    if name == "audio-separator":
        configured = os.environ.get("VOICE_FILTER_AUDIO_SEPARATOR")
        if configured:
            return [configured]
        found = shutil.which("audio-separator")
        if found:
            return [found]
        uvr_python = os.environ.get("VOICE_FILTER_UVR_PYTHON")
        if uvr_python and Path(uvr_python).exists():
            script = Path(uvr_python).parent / "audio-separator.exe"
            if script.exists():
                return [str(script)]
            return [uvr_python, "-m", "audio_separator"]
    if name == "demucs":
        configured = os.environ.get("VOICE_FILTER_DEMUCS")
        if configured:
            return [configured]
        found = shutil.which("demucs")
        if found:
            return [found]
        uvr_python = os.environ.get("VOICE_FILTER_UVR_PYTHON")
        if uvr_python and Path(uvr_python).exists():
            script = Path(uvr_python).parent / "demucs.exe"
            if script.exists():
                return [str(script)]
            return [uvr_python, "-m", "demucs"]
    return None


def separate_vocal(src_wav: Path, dst_wav: Path, backend: str = "auto") -> tuple[bool, str | None, str | None]:
    selected = detect_backend(backend)
    if selected is None:
        return False, None, "uvr_backend_missing"
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    if selected == "audio-separator":
        return _audio_separator(src_wav, dst_wav), selected, None
    if selected == "demucs":
        return _demucs(src_wav, dst_wav), selected, None
    return False, selected, f"unsupported_uvr_backend:{selected}"


def _audio_separator(src_wav: Path, dst_wav: Path) -> bool:
    command = _backend_command("audio-separator")
    if command is None:
        raise RuntimeError("audio-separator command not available")
    out_dir = dst_wav.parent / "_separator_tmp" / dst_wav.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command + [
        str(src_wav),
        "--output_dir",
        str(out_dir),
        "--output_format",
        "WAV",
    ]
    model = os.environ.get("VOICE_FILTER_UVR_MODEL")
    if model:
        cmd.extend(["--model_filename", model])
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "audio-separator failed")
    candidates = sorted(out_dir.glob("*Vocals*.wav")) or sorted(out_dir.glob("*vocals*.wav"))
    if not candidates:
        raise RuntimeError("audio-separator produced no vocals wav")
    decode_to_wav(candidates[0], dst_wav)
    shutil.rmtree(out_dir, ignore_errors=True)
    return True


def _demucs(src_wav: Path, dst_wav: Path) -> bool:
    command = _backend_command("demucs")
    if command is None:
        raise RuntimeError("demucs command not available")
    out_dir = dst_wav.parent / "_demucs_tmp" / dst_wav.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command + ["--two-stems", "vocals", "-o", str(out_dir), str(src_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "demucs failed")
    candidates = sorted(out_dir.rglob("vocals.wav"))
    if not candidates:
        raise RuntimeError("demucs produced no vocals.wav")
    decode_to_wav(candidates[0], dst_wav)
    shutil.rmtree(out_dir, ignore_errors=True)
    return True
