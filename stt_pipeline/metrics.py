"""[10] 메트릭 계산기

원본: Kaggle 노트북 audio.ipynb의 "[10]" 섹션을 모듈로 분리했습니다.
"""

from typing import Dict

from .utils import calculate_cer, calculate_wer


class MetricsCalculator:
    """성능 메트릭 계산"""

    @staticmethod
    def calculate_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
        """종합 메트릭 계산"""
        if not reference or not hypothesis:
            return {
                "wer": None,
                "cer": None,
                "word_accuracy": None
            }

        wer = calculate_wer(reference, hypothesis)
        cer = calculate_cer(reference, hypothesis)

        return {
            "wer": wer,
            "cer": cer,
            "word_accuracy": max(0, 100 - wer)
        }

    @staticmethod
    def calculate_improvement(baseline_wer: float, denoised_wer: float) -> float:
        """WER 개선량 계산 (양수 = 개선)"""
        if baseline_wer is None or denoised_wer is None:
            return None
        return baseline_wer - denoised_wer
