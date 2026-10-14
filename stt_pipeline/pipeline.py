"""[11] 파이프라인 V6

원본: Kaggle 노트북 audio.ipynb의 "[11]" 섹션을 모듈로 분리했습니다.
"""

import os
from queue import Queue
from threading import Thread
from typing import Dict, List

import torch

from .chunk_manager import ChunkManager
from .chunk_merger import ChunkMerger
from .config import config, logger
from .denoiser import DenoiserV6
from .metrics import MetricsCalculator
from .srt_parser import SRTParser
from .stt import WhisperModel
from .utils import calculate_cer, calculate_wer


class PipelineV6:
    """통합 파이프라인 V6 - 초보수적 SNR 기반 스킵"""

    def __init__(self):
        self.whisper_baseline = WhisperModel(config.WHISPER_DEVICE_0)
        self.whisper_denoised = WhisperModel(config.WHISPER_DEVICE_1)
        self.denoiser = DenoiserV6()
        self.comparisons = []

    def process_file(self, audio_path: str, srt_path: str = None) -> Dict:
        """단일 파일 처리"""
        filename = os.path.basename(audio_path)
        logger.info(f"처리 시작: {filename}")

        # SRT 로드
        srt_segments = []
        if srt_path and os.path.exists(srt_path):
            srt_segments = SRTParser.parse_srt(srt_path)
            logger.info(f"SRT 로드: {len(srt_segments)}개 세그먼트")

        # 청크 생성
        chunk_mgr = ChunkManager(audio_path, srt_segments)
        logger.info(f"청크 수: {len(chunk_mgr.chunks)}, SRT 세그먼트: {len(srt_segments)}")

        # 청크별 처리
        chunk_results = []

        if config.ENABLE_PARALLEL and torch.cuda.device_count() >= 2:
            logger.info("병렬 처리 모드 (GPU 2개)")
            chunk_results = self._process_parallel(chunk_mgr, filename)
        else:
            logger.info("순차 처리 모드")
            chunk_results = self._process_sequential(chunk_mgr, filename)

        # 결과 병합
        baseline_texts = [c["baseline_text"] for c in chunk_results]
        denoised_texts = [c["denoised_text"] for c in chunk_results]

        merged_baseline = ChunkMerger.merge_chunks([{"text": t} for t in baseline_texts])
        merged_denoised = ChunkMerger.merge_chunks([{"text": t} for t in denoised_texts])

        # 전체 파일 GT 및 메트릭 계산
        full_gt = chunk_mgr.get_full_gt()

        if full_gt:
            baseline_metrics = MetricsCalculator.calculate_metrics(full_gt, merged_baseline)
            denoised_metrics = MetricsCalculator.calculate_metrics(full_gt, merged_denoised)
            improvement = MetricsCalculator.calculate_improvement(
                baseline_metrics["wer"], denoised_metrics["wer"]
            )
        else:
            baseline_metrics = {"wer": None, "cer": None}
            denoised_metrics = {"wer": None, "cer": None}
            improvement = None

        # 스킵 통계
        skip_count = sum(1 for c in chunk_results if c.get("skipped", False))

        result = {
            "filename": filename,
            "duration": chunk_mgr.duration,
            "n_chunks": len(chunk_mgr.chunks),
            "n_skipped": skip_count,
            "skip_rate": skip_count / len(chunk_mgr.chunks) * 100 if chunk_mgr.chunks else 0,
            "baseline_text": merged_baseline,
            "denoised_text": merged_denoised,
            "ground_truth": full_gt,
            "baseline_wer": baseline_metrics["wer"],
            "baseline_cer": baseline_metrics.get("cer"),
            "denoised_wer": denoised_metrics["wer"],
            "denoised_cer": denoised_metrics.get("cer"),
            "wer_improvement": improvement,
            "chunk_results": chunk_results
        }

        if baseline_metrics["wer"] is not None:
            logger.info(f"결과 - Baseline WER: {baseline_metrics['wer']:.2f}%, "
                        f"Denoised WER: {denoised_metrics['wer']:.2f}%, "
                        f"개선: {improvement:+.2f}%p, 스킵: {skip_count}/{len(chunk_mgr.chunks)}")

        return result

    def _process_parallel(self, chunk_mgr: ChunkManager, filename: str) -> List[Dict]:
        """병렬 처리"""
        logger.info("[병렬] Baseline(GPU0) + Denoise(CPU) + Denoised(GPU1)")

        results = [None] * len(chunk_mgr.chunks)

        baseline_queue = Queue()
        denoised_queue = Queue()

        def baseline_worker():
            for i, chunk in enumerate(chunk_mgr.chunks):
                chunk_path = chunk_mgr.save_chunk(i, "_baseline")
                text, info = self.whisper_baseline.transcribe(chunk_path)
                baseline_queue.put((i, text, info))
                try:
                    os.remove(chunk_path)
                except:
                    pass

        def denoise_worker():
            for i, chunk in enumerate(chunk_mgr.chunks):
                chunk_path = chunk_mgr.save_chunk(i, "_denoise")
                denoised_path, method, analysis = self.denoiser.denoise(chunk_path)
                text, info = self.whisper_denoised.transcribe(denoised_path)
                denoised_queue.put((i, text, method, analysis, info))
                try:
                    os.remove(chunk_path)
                    os.remove(denoised_path)
                except:
                    pass

        baseline_thread = Thread(target=baseline_worker)
        denoised_thread = Thread(target=denoise_worker)

        baseline_thread.start()
        denoised_thread.start()

        baseline_results = {}
        denoised_results = {}

        baseline_thread.join()
        denoised_thread.join()

        while not baseline_queue.empty():
            i, text, info = baseline_queue.get()
            baseline_results[i] = (text, info)

        while not denoised_queue.empty():
            i, text, method, analysis, info = denoised_queue.get()
            denoised_results[i] = (text, method, analysis, info)

        for i, chunk in enumerate(chunk_mgr.chunks):
            baseline_text, baseline_info = baseline_results.get(i, ("", {}))
            denoised_text, method, analysis, denoised_info = denoised_results.get(i, ("", "", {}, {}))

            chunk_gt = chunk.get("ground_truth", "")

            if chunk_gt:
                baseline_wer = calculate_wer(chunk_gt, baseline_text)
                baseline_cer = calculate_cer(chunk_gt, baseline_text)
                denoised_wer = calculate_wer(chunk_gt, denoised_text)
                denoised_cer = calculate_cer(chunk_gt, denoised_text)
                wer_imp = baseline_wer - denoised_wer
            else:
                baseline_wer = baseline_cer = denoised_wer = denoised_cer = wer_imp = None

            result = {
                "idx": i,
                "filename": filename,
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "baseline_text": baseline_text,
                "denoised_text": denoised_text,
                "chunk_ground_truth": chunk_gt,
                "baseline_wer": baseline_wer,
                "baseline_cer": baseline_cer,
                "denoised_wer": denoised_wer,
                "denoised_cer": denoised_cer,
                "wer_improvement": wer_imp,
                "denoise_method": method,
                "snr_db": analysis.get("snr_db", 0),
                "noise_type": analysis.get("noise_type", "unknown"),
                "skipped": analysis.get("skipped", False),
                "skip_reason": analysis.get("skip_reason", "")
            }

            results[i] = result
            self.comparisons.append(result)

        return results

    def _process_sequential(self, chunk_mgr: ChunkManager, filename: str) -> List[Dict]:
        """순차 처리"""
        results = []

        for i, chunk in enumerate(chunk_mgr.chunks):
            chunk_path = chunk_mgr.save_chunk(i)

            baseline_text, _ = self.whisper_baseline.transcribe(chunk_path)
            denoised_path, method, analysis = self.denoiser.denoise(chunk_path)
            denoised_text, _ = self.whisper_baseline.transcribe(denoised_path)

            chunk_gt = chunk.get("ground_truth", "")

            if chunk_gt:
                baseline_wer = calculate_wer(chunk_gt, baseline_text)
                baseline_cer = calculate_cer(chunk_gt, baseline_text)
                denoised_wer = calculate_wer(chunk_gt, denoised_text)
                denoised_cer = calculate_cer(chunk_gt, denoised_text)
                wer_imp = baseline_wer - denoised_wer
            else:
                baseline_wer = baseline_cer = denoised_wer = denoised_cer = wer_imp = None

            result = {
                "idx": i,
                "filename": filename,
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "baseline_text": baseline_text,
                "denoised_text": denoised_text,
                "chunk_ground_truth": chunk_gt,
                "baseline_wer": baseline_wer,
                "baseline_cer": baseline_cer,
                "denoised_wer": denoised_wer,
                "denoised_cer": denoised_cer,
                "wer_improvement": wer_imp,
                "denoise_method": method,
                "snr_db": analysis.get("snr_db", 0),
                "noise_type": analysis.get("noise_type", "unknown"),
                "skipped": analysis.get("skipped", False),
                "skip_reason": analysis.get("skip_reason", "")
            }

            results.append(result)
            self.comparisons.append(result)

            try:
                os.remove(chunk_path)
                os.remove(denoised_path)
            except:
                pass

        return results

    def unload(self):
        """리소스 해제"""
        self.whisper_baseline.unload()
        self.whisper_denoised.unload()
        self.denoiser.unload()
