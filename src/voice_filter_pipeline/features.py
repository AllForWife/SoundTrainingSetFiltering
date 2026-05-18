from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import dbfs, rms
from .config import Thresholds


@dataclass(frozen=True)
class AudioFeatures:
    duration_sec: float
    speech_ratio: float
    silence_ratio: float
    clipping_ratio: float
    rms_dbfs: float
    noise_dbfs: float
    snr_db: float
    bandwidth_score: float
    median_f0_hz: float | None


def frame_rms(wav: np.ndarray, sr: int, frame_ms: float = 30.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000.0))
    if wav.size < frame:
        return np.array([rms(wav)], dtype=np.float32)
    usable = wav[: (wav.size // frame) * frame]
    frames = usable.reshape(-1, frame)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)).astype(np.float32)


def speech_mask(wav: np.ndarray, sr: int) -> np.ndarray:
    levels = frame_rms(wav, sr)
    if levels.size == 0:
        return np.array([], dtype=bool)
    floor = float(np.percentile(levels, 20))
    ceiling = float(np.percentile(levels, 95))
    if ceiling > 0.01 and floor > 0 and ceiling / floor < 1.5:
        return levels > max(0.003, floor * 0.5)
    threshold = max(floor * 2.6, ceiling * 0.08, 0.003)
    return levels > threshold


def estimate_f0(wav: np.ndarray, sr: int, mask: np.ndarray) -> float | None:
    try:
        librosa = __import__("librosa")
    except Exception:
        return estimate_f0_autocorr(wav, sr)
    if wav.size < sr // 2:
        return None
    try:
        f0, _, _ = librosa.pyin(
            wav.astype(np.float64),
            fmin=70,
            fmax=420,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
    except Exception:
        return None
    values = f0[np.isfinite(f0)]
    if values.size < 3:
        return estimate_f0_autocorr(wav, sr)
    return float(np.median(values))


def estimate_f0_autocorr(wav: np.ndarray, sr: int) -> float | None:
    if wav.size < sr // 4:
        return None
    centered = wav.astype(np.float64) - float(np.mean(wav))
    if not np.any(centered):
        return None
    max_len = min(centered.size, sr * 2)
    centered = centered[:max_len]
    corr = np.correlate(centered, centered, mode="full")[max_len - 1 :]
    min_lag = max(1, int(sr / 420))
    max_lag = min(corr.size - 1, int(sr / 70))
    if max_lag <= min_lag:
        return None
    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    if lag <= 0:
        return None
    confidence = corr[lag] / max(corr[0], 1e-12)
    if confidence < 0.25:
        return None
    return float(sr / lag)


def bandwidth_score(wav: np.ndarray, sr: int) -> float:
    if wav.size < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(wav))
    if not np.any(spectrum):
        return 0.0
    freqs = np.fft.rfftfreq(wav.size, 1.0 / sr)
    total = float(np.sum(spectrum))
    high = float(np.sum(spectrum[(freqs >= 3000) & (freqs <= min(9000, sr / 2))]))
    very_low = float(np.sum(spectrum[freqs < 80]))
    return max(0.0, min(1.0, (high / total) * 7.0 - (very_low / total) * 3.0))


def extract_features(wav: np.ndarray, sr: int) -> AudioFeatures:
    duration = wav.size / float(sr) if sr else 0.0
    levels = frame_rms(wav, sr)
    mask = speech_mask(wav, sr)
    speech_ratio = float(np.mean(mask)) if mask.size else 0.0
    silence_ratio = 1.0 - speech_ratio
    clipping_ratio = float(np.mean(np.abs(wav) >= 0.98)) if wav.size else 0.0
    low = float(np.percentile(levels, 20)) if levels.size else 0.0
    speech_levels = levels[mask] if mask.size and np.any(mask) else levels
    speech_level = float(np.percentile(speech_levels, 80)) if speech_levels.size else 0.0
    snr = dbfs(speech_level) - dbfs(low)
    return AudioFeatures(
        duration_sec=duration,
        speech_ratio=speech_ratio,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        rms_dbfs=dbfs(rms(wav)),
        noise_dbfs=dbfs(low),
        snr_db=snr,
        bandwidth_score=bandwidth_score(wav, sr),
        median_f0_hz=estimate_f0(wav, sr, mask),
    )


def classify_gender(features: AudioFeatures, thresholds: Thresholds) -> tuple[str, float, str | None]:
    if features.duration_sec < thresholds.min_duration_sec:
        return "unknown", 0.0, "too_short"
    if features.speech_ratio < thresholds.min_speech_ratio:
        return "unknown", 0.0, "no_voice"
    if features.median_f0_hz is None:
        return "unknown", 0.35, "f0_unavailable"
    f0 = features.median_f0_hz
    if f0 >= thresholds.female_f0_hz:
        confidence = min(0.98, 0.60 + (f0 - thresholds.female_f0_hz) / 120.0)
        return "female", confidence, None
    if f0 <= thresholds.male_f0_hz:
        confidence = min(0.98, 0.60 + (thresholds.male_f0_hz - f0) / 90.0)
        return "male", confidence, None
    return "unknown", 0.45, "gender_borderline"


def quality_score(features: AudioFeatures, thresholds: Thresholds) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 1.0
    if features.duration_sec < thresholds.min_duration_sec:
        score -= 0.45
        reasons.append("too_short")
    if features.speech_ratio < thresholds.min_speech_ratio:
        score -= 0.45
        reasons.append("low_speech_ratio")
    if features.silence_ratio > thresholds.max_silence_ratio:
        score -= 0.25
        reasons.append("too_much_silence")
    if features.clipping_ratio > thresholds.max_clipping_ratio:
        score -= min(0.40, features.clipping_ratio * 80)
        reasons.append("clipping")
    if features.snr_db < 10:
        score -= 0.30
        reasons.append("low_snr")
    elif features.snr_db < 16:
        score -= 0.15
        reasons.append("borderline_snr")
    if features.rms_dbfs < -38:
        score -= 0.20
        reasons.append("too_quiet")
    if features.rms_dbfs > -5:
        score -= 0.25
        reasons.append("too_loud")
    score -= max(0.0, 0.20 - features.bandwidth_score * 0.20)
    if features.bandwidth_score < 0.15:
        reasons.append("narrow_bandwidth")
    return max(0.0, min(1.0, round(score, 4))), reasons
