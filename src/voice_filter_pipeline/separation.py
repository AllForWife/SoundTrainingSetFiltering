from __future__ import annotations

import json
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


def separate_vocals_batch(
    src_files: list[Path],
    out_dir: Path,
    backend: str = "auto",
) -> tuple[dict[Path, Path], str | None, dict[Path, str]]:
    selected = detect_backend(backend)
    if selected is None:
        return {}, None, {src.resolve(): "uvr_backend_missing" for src in src_files}
    if not src_files:
        return {}, selected, {}
    if selected != "audio-separator":
        return _separate_vocals_batch_fallback(src_files, out_dir, selected)
    return _audio_separator_batch(src_files, out_dir), selected, {}


def _audio_separator(src_wav: Path, dst_wav: Path) -> bool:
    command = _backend_command("audio-separator")
    if command is None:
        raise RuntimeError("audio-separator command not available")
    _ensure_gpu_memory_available()
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
    proc = _run_backend(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "audio-separator failed")
    candidates = sorted(out_dir.glob("*Vocals*.wav")) or sorted(out_dir.glob("*vocals*.wav"))
    if not candidates:
        raise RuntimeError("audio-separator produced no vocals wav")
    decode_to_wav(candidates[0], dst_wav)
    shutil.rmtree(out_dir, ignore_errors=True)
    return True


def _audio_separator_batch(src_files: list[Path], out_dir: Path) -> dict[Path, Path]:
    command = _backend_command("audio-separator")
    if command is None:
        raise RuntimeError("audio-separator command not available")
    _ensure_gpu_memory_available()
    out_dir.mkdir(parents=True, exist_ok=True)
    src_files = [src.resolve() for src in src_files]
    cmd = command + [str(src) for src in src_files] + [
        "--output_dir",
        str(out_dir),
        "--output_format",
        "WAV",
        "--single_stem",
        "Vocals",
        "--sample_rate",
        "24000",
    ]
    model = os.environ.get("VOICE_FILTER_UVR_MODEL")
    if model:
        cmd.extend(["--model_filename", model])
    model_dir = os.environ.get("VOICE_FILTER_UVR_MODEL_DIR")
    if model_dir:
        cmd.extend(["--model_file_dir", model_dir])
    mdx_batch_size = os.environ.get("VOICE_FILTER_UVR_MDX_BATCH_SIZE")
    if mdx_batch_size:
        cmd.extend(["--mdx_batch_size", mdx_batch_size])
    mdxc_batch_size = os.environ.get("VOICE_FILTER_UVR_MDXC_BATCH_SIZE")
    if mdxc_batch_size:
        cmd.extend(["--mdxc_batch_size", mdxc_batch_size])
    if os.environ.get("VOICE_FILTER_UVR_USE_AUTOCAST", "1").strip().lower() not in {"0", "false", "no"}:
        cmd.append("--use_autocast")
    custom_names = _custom_output_names(src_files)
    if custom_names:
        cmd.extend(["--custom_output_names", json.dumps(custom_names, ensure_ascii=False)])
    proc = _run_backend(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "audio-separator batch failed")
    return _collect_batch_outputs(src_files, out_dir)


def _custom_output_names(src_files: list[Path]) -> dict[str, str] | None:
    # The CLI accepts stem names per source. Unique stems avoid ambiguous batch output names.
    stems = [src.stem for src in src_files]
    if len(set(stems)) != len(stems):
        return None
    return {stem: stem for stem in stems}


def _collect_batch_outputs(src_files: list[Path], out_dir: Path) -> dict[Path, Path]:
    wavs = sorted(out_dir.rglob("*.wav"))
    results: dict[Path, Path] = {}
    for src in src_files:
        src = src.resolve()
        matches = [
            path
            for path in wavs
            if src.stem.lower() in path.stem.lower()
            and ("vocal" in path.stem.lower() or "vocals" in path.stem.lower())
        ]
        if not matches:
            matches = [path for path in wavs if src.stem.lower() in path.stem.lower()]
        if matches:
            results[src] = matches[0]
    return results


def _separate_vocals_batch_fallback(
    src_files: list[Path],
    out_dir: Path,
    selected: str,
) -> tuple[dict[Path, Path], str, dict[Path, str]]:
    results: dict[Path, Path] = {}
    errors: dict[Path, str] = {}
    for src in src_files:
        tmp = out_dir / f"{src.stem}.wav"
        try:
            ok, _, error = separate_vocal(src, tmp, selected)
            if ok:
                results[src.resolve()] = tmp
            else:
                errors[src.resolve()] = error or "uvr_failed"
        except Exception as exc:
            errors[src.resolve()] = f"uvr_failed:{exc}"
    return results, selected, errors


def _demucs(src_wav: Path, dst_wav: Path) -> bool:
    command = _backend_command("demucs")
    if command is None:
        raise RuntimeError("demucs command not available")
    out_dir = dst_wav.parent / "_demucs_tmp" / dst_wav.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command + ["--two-stems", "vocals", "-o", str(out_dir), str(src_wav)]
    proc = _run_backend(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "demucs failed")
    candidates = sorted(out_dir.rglob("vocals.wav"))
    if not candidates:
        raise RuntimeError("demucs produced no vocals.wav")
    decode_to_wav(candidates[0], dst_wav)
    shutil.rmtree(out_dir, ignore_errors=True)
    return True


def _run_backend(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    timeout = int(os.environ.get("VOICE_FILTER_UVR_TIMEOUT_SEC", "1800"))
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"uvr_timeout_after_{timeout}s") from exc


def _ensure_gpu_memory_available() -> None:
    raw = os.environ.get("VOICE_FILTER_UVR_MIN_FREE_MB", "6000")
    try:
        required_mb = int(raw)
    except ValueError:
        required_mb = 6000
    if required_mb <= 0:
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return
    proc = subprocess.run(
        [nvidia_smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    if proc.returncode != 0:
        return
    values = [int(part.strip()) for part in proc.stdout.splitlines() if part.strip().isdigit()]
    if values and max(values) < required_mb:
        raise RuntimeError(f"uvr_gpu_memory_low:{max(values)}MB_free<{required_mb}MB_required")
