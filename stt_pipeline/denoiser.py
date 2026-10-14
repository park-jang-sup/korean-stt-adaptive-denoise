"""[8] 디노이저 V6 (SNR 기반 조건부 노이즈 제거)

원본: Kaggle 노트북 audio.ipynb의 "[8]" 섹션을 모듈로 분리했습니다.
"""

from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf

from .config import config, logger
from .noise_analyzer import NoiseAnalyzer


class DenoiserV6:
    """V6: 초보수적 SNR 기반 노이즈 제거 (SNR >= 6dB 스킵)"""

    def __init__(self):
        self._nr = None
        self.skip_count = 0
        self.process_count = 0

    def _load_noisereduce(self):
        if self._nr is None:
            import noisereduce as nr
            self._nr = nr
            logger.info("noisereduce 로딩 완료")

    def denoise(self, audio_path: str) -> Tuple[str, str, Dict]:
        """
        V6 핵심: 초보수적 SNR 기반 노이즈 제거
        - SNR >= 6dB: 스킵 (원본 그대로) - 가장 보수적
        - SNR 4-6dB: 가벼운 처리
        - SNR < 4dB: 다단계 처리
        """
        self._load_noisereduce()

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 노이즈 레벨 분석
        _, category, analysis = NoiseAnalyzer.estimate_noise_level(audio, sr)
        snr_db = analysis["snr_db"]

        # V6 핵심: SNR >= 6dB 스킵 (초보수적)
        if snr_db >= config.SNR_SKIP_THRESHOLD:
            # SNR >= 6dB - 스킵 (원본 그대로 사용)
            self.skip_count += 1
            output_path = audio_path.replace(".wav", "_denoised.wav")
            sf.write(output_path, audio, sr)  # 원본 그대로 저장
            analysis["skipped"] = True
            analysis["skip_reason"] = f"SNR {snr_db:.1f}dB >= {config.SNR_SKIP_THRESHOLD}dB"
            return output_path, "skipped", analysis

        # 노이즈 제거 수행 (SNR < 6dB에서만)
        self.process_count += 1
        analysis["skipped"] = False

        if snr_db >= config.SNR_NOISY_THRESHOLD:
            # SNR 4-6dB: 가벼운 처리
            denoised, method = self._process_light(audio, sr, analysis)
        else:
            # SNR < 4dB: 다단계 처리
            denoised, method = self._process_noisy(audio, sr, analysis)

        output_path = audio_path.replace(".wav", "_denoised.wav")
        sf.write(output_path, denoised, sr)

        return output_path, method, analysis

    def _process_light(self, audio: np.ndarray, sr: int, analysis: Dict) -> Tuple[np.ndarray, str]:
        """가벼운 처리 (SNR 4-6dB)"""
        noise_type = analysis.get("noise_type", "non-stationary")

        if noise_type == "stationary":
            denoised = self._nr.reduce_noise(
                y=audio, sr=sr,
                stationary=True,
                prop_decrease=0.3,
                n_fft=2048, hop_length=512
            )
        else:
            denoised = self._nr.reduce_noise(
                y=audio, sr=sr,
                stationary=False,
                prop_decrease=0.3,
                n_fft=2048, hop_length=512,
                time_constant_s=2.0
            )
        return denoised, f"moderate_{noise_type}"

    def _process_noisy(self, audio: np.ndarray, sr: int, analysis: Dict) -> Tuple[np.ndarray, str]:
        """다단계 처리 (SNR < 4dB)"""
        noise_type = analysis.get("noise_type", "non-stationary")
        snr_db = analysis.get("snr_db", 5.0)

        # Stage 1: Stationary 노이즈 제거
        stage1 = self._nr.reduce_noise(
            y=audio, sr=sr,
            stationary=True,
            prop_decrease=0.5,
            n_fft=2048, hop_length=512
        )

        # Stage 2: 노이즈 유형별 추가 처리
        if noise_type == "impulsive":
            stage2 = self._nr.reduce_noise(
                y=stage1, sr=sr,
                stationary=False,
                prop_decrease=0.3,
                n_fft=512, hop_length=128
            )
        elif noise_type == "non-stationary":
            stage2 = self._nr.reduce_noise(
                y=stage1, sr=sr,
                stationary=False,
                prop_decrease=0.4,
                n_fft=1024, hop_length=256,
                time_constant_s=1.5
            )
        else:
            stage2 = self._nr.reduce_noise(
                y=stage1, sr=sr,
                stationary=True,
                prop_decrease=0.3,
                n_fft=2048, hop_length=512
            )

        # Stage 3: 매우 낮은 SNR일 때만 추가 Wiener
        if snr_db < 6.0:
            stage3 = self._wiener_filter(stage2, sr, strength=0.2)
        else:
            stage3 = stage2

        return stage3, f"multi_stage_{noise_type}"

    def _wiener_filter(self, audio: np.ndarray, sr: int, strength: float = 0.3) -> np.ndarray:
        """가벼운 Wiener 필터"""
        n_fft = 2048
        hop_length = 512

        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(S), np.angle(S)

        noise_frames = min(10, mag.shape[1])
        noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

        snr_est = (mag ** 2) / (noise_mag ** 2 + 1e-10)
        gain = snr_est / (1 + snr_est)
        gain = gain * strength + (1 - strength)

        S_filtered = (mag * gain) * np.exp(1j * phase)
        return librosa.istft(S_filtered, hop_length=hop_length, length=len(audio))

    def get_stats(self) -> Dict:
        """스킵/처리 통계 반환"""
        total = self.skip_count + self.process_count
        return {
            "total_chunks": total,
            "skipped": self.skip_count,
            "processed": self.process_count,
            "skip_rate": self.skip_count / total * 100 if total > 0 else 0
        }

    def unload(self):
        self._nr = None
        logger.info(f"Denoiser V6 언로드됨 (스킵: {self.skip_count}, 처리: {self.process_count})")
