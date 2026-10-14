"""[13] 메인 실행

원본: Kaggle 노트북 audio.ipynb의 "[13]" 섹션을 모듈로 분리했습니다.
"""

import glob
import json
import os
from typing import List, Optional, Tuple

import torch

from .config import config, logger
from .pipeline import PipelineV6
from .report import ReportGeneratorV6
from .srt_parser import SRTParser


def find_audio_files(data_dir: str) -> List[Tuple[str, Optional[str]]]:
    """오디오 파일과 대응 SRT 찾기"""
    audio_files = []

    patterns = ["**/*.wav", "**/*.WAV", "**/*.mp3", "**/*.flac"]

    for pattern in patterns:
        for audio_path in glob.glob(os.path.join(data_dir, pattern), recursive=True):
            srt_path = SRTParser.find_srt_file(audio_path)
            audio_files.append((audio_path, srt_path))

    return audio_files


def main():
    """메인 실행"""
    logger.info("=" * 60)
    logger.info("한국어 STT 통합 파이프라인 V6")
    logger.info("(SNR 기반 적응적 노이즈 제거)")
    logger.info(f"스킵 임계값: SNR ≥ {config.SNR_SKIP_THRESHOLD}dB")
    logger.info("=" * 60)

    # GPU 정보
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

    # 데이터 찾기
    data_dirs = [
        "/kaggle/input",
        "/content/drive/MyDrive",
        "./data",
        "."
    ]

    audio_files = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            audio_files = find_audio_files(data_dir)
            if audio_files:
                break

    if not audio_files:
        logger.error("오디오 파일을 찾을 수 없습니다.")
        return

    logger.info(f"발견된 파일: {len(audio_files)}개")

    # SRT 매칭 확인
    srt_matched = sum(1 for _, srt in audio_files if srt is not None)
    logger.info(f"SRT 매칭: {srt_matched}/{len(audio_files)}개")

    # 파이프라인 실행
    pipeline = PipelineV6()
    results = []

    try:
        for audio_path, srt_path in audio_files:
            logger.info("=" * 60)
            result = pipeline.process_file(audio_path, srt_path)
            results.append(result)

        logger.info("=" * 60)
        logger.info("리포트 생성 중...")

        # 리포트 생성
        ReportGeneratorV6.generate_html_report(
            results,
            pipeline.comparisons,
            f"{config.OUTPUT_DIR}/comparison_report_v6.html",
            pipeline.denoiser.get_stats()
        )

        ReportGeneratorV6.generate_csv(
            pipeline.comparisons,
            f"{config.OUTPUT_DIR}/detailed_results_v6.csv"
        )

        stats = ReportGeneratorV6.generate_statistics(
            results,
            pipeline.comparisons,
            f"{config.OUTPUT_DIR}/statistics_v6.json",
            pipeline.denoiser.get_stats()
        )

        # 최종 결과 출력
        logger.info("=" * 60)
        logger.info("📊 최종 결과 (V6 - 초보수적 스킵)")
        logger.info("=" * 60)

        if stats:
            fl = stats.get("file_level", {})
            cl = stats.get("chunk_level", {})
            ss = stats.get("skip_statistics", {})

            logger.info(f"  처리 파일: {fl.get('n_files', 0)}개")
            logger.info(f"  [스킵 통계]")
            logger.info(
                f"    스킵된 청크: {ss.get('skipped_chunks', 0)}/{ss.get('total_chunks', 0)} ({ss.get('skip_rate', 0):.1f}%)")
            logger.info(f"    스킵 청크 평균 SNR: {ss.get('skipped_avg_snr', 0):.1f}dB")
            logger.info(f"    처리 청크 평균 SNR: {ss.get('processed_avg_snr', 0):.1f}dB")
            logger.info(f"  [파일 단위]")
            logger.info(
                f"    Baseline WER: {fl.get('baseline_wer_mean', 0):.2f}% ± {fl.get('baseline_wer_std', 0):.2f}")
            logger.info(
                f"    Denoised WER: {fl.get('denoised_wer_mean', 0):.2f}% ± {fl.get('denoised_wer_std', 0):.2f}")
            logger.info(f"    평균 WER 개선: {fl.get('mean_improvement', 0):+.2f}%p")
            logger.info(f"    개선된 파일: {fl.get('improved_files', 0)}/{fl.get('n_files', 0)}")
            logger.info(f"  [청크 단위]")
            logger.info(f"    유효 청크: {cl.get('n_chunks', 0)}개")
            logger.info(f"    Baseline WER: {cl.get('baseline_wer_mean', 0):.2f}%")
            logger.info(f"    Denoised WER: {cl.get('denoised_wer_mean', 0):.2f}%")
            logger.info(f"    개선된 청크: {cl.get('improved_chunks', 0)}/{cl.get('n_chunks', 0)}")

    finally:
        pipeline.unload()
        logger.info("파이프라인 V6 종료")

    print(f"\n[INFO] ✅ V6 완료!")

    # 통계 출력
    if stats:
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
