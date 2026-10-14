"""[3] 유틸리티 함수 (WER/CER 계산, GPU 메모리 정리)

원본: Kaggle 노트북 audio.ipynb의 "[3]" 섹션을 모듈로 분리했습니다.
"""

import gc
import re

import jiwer
import torch


def clear_gpu_memory():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def normalize_text(text: str) -> str:
    """텍스트 정규화 (한국어 특화)"""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'[.,!?;:"\'-]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def calculate_wer(reference: str, hypothesis: str) -> float:
    """WER 계산 (정규화 포함)"""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return 0.0 if not hyp_norm else 100.0

    try:
        wer = jiwer.wer(ref_norm, hyp_norm)
        return min(wer * 100, 100.0)
    except Exception:
        return 100.0


def calculate_cer(reference: str, hypothesis: str) -> float:
    """CER 계산 (문자 단위)"""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return 0.0 if not hyp_norm else 100.0

    try:
        cer = jiwer.cer(ref_norm, hyp_norm)
        return min(cer * 100, 100.0)
    except Exception:
        return 100.0
