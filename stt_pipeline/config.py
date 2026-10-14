"""[2] 설정 (V6 - SNR 기반 스킵 임계값)

원본: Kaggle 노트북 audio.ipynb의 "[2]" 섹션을 모듈로 분리했습니다.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict

import torch

from .logging_utils import setup_logging


@dataclass
class Config:
    """파이프라인 설정"""
    # GPU 할당
    WHISPER_DEVICE_0: str = "cuda:0"
    WHISPER_DEVICE_1: str = "cuda:1"

    # 오디오 설정
    SAMPLE_RATE: int = 16000
    CHUNK_DURATION: int = 30
    OVERLAP_SEC: float = 3.0

    # V6 핵심: SNR 기반 스킵 임계값 (초보수적)
    SNR_SKIP_THRESHOLD: float = 6.0  # V5: 8.0 → V6: 6.0 (SNR < 6dB에서만 처리)

    # 노이즈 판단 임계값 (SNR dB 기준) - V6에서는 단순화
    SNR_CLEAN_THRESHOLD: float = 18.0
    SNR_MODERATE_THRESHOLD: float = 12.0
    SNR_NOISY_THRESHOLD: float = 4.0  # V6: 6.0 → 4.0 (더 엄격한 다단계 처리)

    # 노이즈 제거 강도 설정
    DENOISE_STRENGTH_CLEAN: float = 0.2
    DENOISE_STRENGTH_MODERATE: float = 0.4
    DENOISE_STRENGTH_NOISY: float = 0.6

    # 스펙트럼 평탄도 임계값
    SPECTRAL_FLATNESS_THRESHOLD: float = 0.15

    # 청크 병합 설정
    MERGE_SIMILARITY_THRESHOLD: float = 0.65
    MERGE_MAX_OVERLAP_WORDS: int = 15

    # SRT 파일 매칭 규칙
    SRT_SUFFIX_MAPPINGS: Dict[str, str] = field(default_factory=lambda: {
        "_NV": "_VN",
        "_VN": "_NV",
        "_noise": "_clean",
        "_noisy": "_clean",
    })

    # 병렬 처리
    ENABLE_PARALLEL: bool = True
    MAX_WORKERS: int = 2

    # 출력 디렉토리
    # 로컬 실행 시 환경 변수 STT_OUTPUT_DIR로 출력 경로를 바꿀 수 있습니다.
    OUTPUT_DIR: str = os.environ.get("STT_OUTPUT_DIR", "/kaggle/working/results_v6")

    def __post_init__(self):
        """설정 검증"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger_temp = logging.getLogger("Config")
            logger_temp.info(f"GPU {gpu_count}개 감지됨")

            if gpu_count < 2:
                logger_temp.warning("GPU 1개만 감지 - 단일 GPU 모드")
                self.WHISPER_DEVICE_1 = "cuda:0"
                self.ENABLE_PARALLEL = False
        else:
            self.WHISPER_DEVICE_0 = "cpu"
            self.WHISPER_DEVICE_1 = "cpu"
            self.ENABLE_PARALLEL = False

        assert 0 < self.OVERLAP_SEC < self.CHUNK_DURATION
        assert 0 < self.MERGE_SIMILARITY_THRESHOLD < 1.0
        assert self.SNR_MODERATE_THRESHOLD < self.SNR_CLEAN_THRESHOLD

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(f"{self.OUTPUT_DIR}/images", exist_ok=True)


# 전역 설정 및 로거
config = Config()
logger = setup_logging(config.OUTPUT_DIR)
