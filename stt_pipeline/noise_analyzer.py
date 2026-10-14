"""[6] 노이즈 분석기 (SNR·스펙트럼 평탄도)

원본: Kaggle 노트북 audio.ipynb의 "[6]" 섹션을 모듈로 분리했습니다.
"""

from typing import Dict, Tuple

import librosa
import numpy as np

from .config import config


class NoiseAnalyzer:
    """노이즈 레벨 추정"""

    @staticmethod
    def estimate_snr(audio: np.ndarray, sr: int) -> float:
        """SNR 추정 (VAD 기반)"""
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)

        threshold = np.percentile(energy, 30)
        speech_frames = energy > threshold
        noise_frames = ~speech_frames

        if np.sum(speech_frames) == 0 or np.sum(noise_frames) == 0:
            return 20.0

        speech_energy = np.mean(energy[speech_frames])
        noise_energy = np.mean(energy[noise_frames]) + 1e-10

        snr_db = 10 * np.log10(speech_energy / noise_energy)
        return np.clip(snr_db, -10, 40)

    @staticmethod
    def estimate_spectral_flatness(audio: np.ndarray, sr: int) -> float:
        """스펙트럼 평탄도 추정"""
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))

        geometric_mean = np.exp(np.mean(np.log(S + 1e-10), axis=0))
        arithmetic_mean = np.mean(S, axis=0)

        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        return float(np.mean(flatness))

    @staticmethod
    def detect_noise_type(audio: np.ndarray, sr: int) -> str:
        """노이즈 유형 감지"""
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)

        if len(energy) < 2:
            return "unknown"

        energy_std = np.std(energy)
        energy_mean = np.mean(energy)
        cv = energy_std / (energy_mean + 1e-10)

        spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        spec_diff = np.diff(spec, axis=1)
        spec_change = np.mean(np.abs(spec_diff))

        if cv < 0.5 and spec_change < 0.1:
            return "stationary"
        elif cv > 1.5:
            return "impulsive"
        else:
            return "non-stationary"

    @staticmethod
    def estimate_noise_level(audio: np.ndarray, sr: int) -> Tuple[float, str, Dict]:
        """종합 노이즈 레벨 추정"""
        snr_db = NoiseAnalyzer.estimate_snr(audio, sr)
        spectral_flatness = NoiseAnalyzer.estimate_spectral_flatness(audio, sr)
        noise_type = NoiseAnalyzer.detect_noise_type(audio, sr)

        details = {
            "snr_db": snr_db,
            "spectral_flatness": spectral_flatness,
            "noise_type": noise_type
        }

        if snr_db > config.SNR_CLEAN_THRESHOLD:
            return 0.2, "clean", details
        elif snr_db > config.SNR_MODERATE_THRESHOLD:
            return 0.5, "moderate", details
        else:
            return 0.8, "noisy", details
