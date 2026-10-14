"""[7] Whisper 모델

원본: Kaggle 노트북 audio.ipynb의 "[7]" 섹션을 모듈로 분리했습니다.
"""

from typing import Dict, Tuple

import numpy as np

from .config import config, logger
from .utils import clear_gpu_memory


class WhisperModel:
    """Faster-Whisper 래퍼"""

    def __init__(self, device: str = None):
        self._model = None
        self._device = device or config.WHISPER_DEVICE_0

    def _load(self):
        if self._model is None:
            logger.info(f"Whisper large-v3 로딩 → {self._device}")
            from faster_whisper import WhisperModel as FW

            device = "cuda" if "cuda" in self._device else "cpu"
            device_index = int(self._device.split(":")[-1]) if "cuda" in self._device else 0

            self._model = FW(
                "large-v3",
                device=device,
                device_index=device_index,
                compute_type="float16" if device == "cuda" else "int8",
                cpu_threads=4,
                num_workers=2
            )
            logger.info(f"Whisper 로딩 완료 ({self._device})")

    def transcribe(self, audio_path: str, language: str = "ko") -> Tuple[str, Dict]:
        self._load()

        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                temperature=[0.0, 0.2, 0.4, 0.6],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 100,
                    "speech_pad_ms": 30,
                },
                condition_on_previous_text=False,
                word_timestamps=False,
            )

            texts = []
            total_prob = 0.0
            count = 0

            for seg in segments:
                texts.append(seg.text.strip())
                if hasattr(seg, 'avg_logprob'):
                    total_prob += seg.avg_logprob
                    count += 1

            avg_confidence = np.exp(total_prob / count) if count > 0 else 0.0

            return " ".join(texts), {
                "language": info.language,
                "language_probability": info.language_probability,
                "avg_confidence": avg_confidence,
                "duration": info.duration
            }
        except Exception as e:
            logger.error(f"Whisper 오류: {e}")
            return "", {"error": str(e)}

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            clear_gpu_memory()
            logger.info("Whisper 모델 언로드됨")
