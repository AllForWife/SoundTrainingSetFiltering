from __future__ import annotations

import numpy as np

from voice_filter_pipeline.config import Thresholds
from voice_filter_pipeline.features import classify_gender, extract_features, quality_score


def sine(freq: float, seconds: float = 1.4, sr: int = 24000, amp: float = 0.2) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_gender_heuristic_identifies_high_f0_as_female() -> None:
    features = extract_features(sine(220), 24000)
    gender, confidence, reason = classify_gender(features, Thresholds())
    assert gender == "female"
    assert confidence >= 0.6
    assert reason is None


def test_gender_heuristic_identifies_low_f0_as_male() -> None:
    features = extract_features(sine(120), 24000)
    gender, confidence, reason = classify_gender(features, Thresholds())
    assert gender == "male"
    assert confidence >= 0.6
    assert reason is None


def test_quality_rejects_silence() -> None:
    features = extract_features(np.zeros(24000, dtype=np.float32), 24000)
    score, reasons = quality_score(features, Thresholds())
    assert score < 0.7
    assert "low_speech_ratio" in reasons
