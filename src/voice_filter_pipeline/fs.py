from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

from .config import AUDIO_EXTENSIONS


def find_audio_files(clips_dir: Path) -> list[Path]:
    if not clips_dir.exists():
        raise FileNotFoundError(f"Missing clips directory: {clips_dir}")
    return sorted(
        path.resolve()
        for path in clips_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def select_files(files: list[Path], limit: int | None, seed: int, randomize: bool) -> list[Path]:
    if limit is None or limit >= len(files):
        return files
    if not randomize:
        return files[:limit]
    rng = random.Random(seed)
    return sorted(rng.sample(files, limit))


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def output_name(row: dict, suffix: str = ".wav") -> str:
    stem = row.get("sha1") or Path(str(row["source_path"])).stem
    return f"{stem}{suffix}"
