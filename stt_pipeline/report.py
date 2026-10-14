"""[12] 리포트 생성기 V6

원본: Kaggle 노트북 audio.ipynb의 "[12]" 섹션을 모듈로 분리했습니다.
"""

import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .config import config, logger

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class ReportGeneratorV6:
    """V6 리포트 생성"""

    @staticmethod
    def generate_html_report(results: List[dict], comparisons: List[dict],
                             output_path: str, denoiser_stats: Dict = None):
        """HTML 리포트 생성"""
        valid_results = [r for r in results if r.get("baseline_wer") is not None]
        valid_comparisons = [c for c in comparisons if c.get("baseline_wer") is not None]

        if valid_results:
            avg_baseline_wer = np.mean([r["baseline_wer"] for r in valid_results])
            avg_denoised_wer = np.mean([r["denoised_wer"] for r in valid_results])
            avg_improvement = np.mean([r["wer_improvement"] for r in valid_results])
            avg_baseline_cer = np.mean([r["baseline_cer"] for r in valid_results if r.get("baseline_cer")])
            avg_denoised_cer = np.mean([r["denoised_cer"] for r in valid_results if r.get("denoised_cer")])
        else:
            avg_baseline_wer = avg_denoised_wer = avg_improvement = 0
            avg_baseline_cer = avg_denoised_cer = 0

        # 스킵 통계
        total_skipped = sum(r.get("n_skipped", 0) for r in results)
        total_chunks = sum(r.get("n_chunks", 0) for r in results)
        skip_rate = total_skipped / total_chunks * 100 if total_chunks > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>노이즈 제거 STT 비교 리포트 V6</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 2em; }}
        .header .subtitle {{ opacity: 0.9; margin-top: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #1a237e; }}
        .stat-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        .skip-card {{ background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); }}
        .skip-value {{ color: #2e7d32 !important; }}
        .chunk-card {{ background: white; margin: 15px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .chunk-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #eee; flex-wrap: wrap; gap: 10px; }}
        .metrics-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .method-badge {{ background: #e3f2fd; color: #1565c0; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; }}
        .skip-badge {{ background: #e8f5e9; color: #2e7d32; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: bold; }}
        .snr-badge {{ background: #fff3e0; color: #ef6c00; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; }}
        .wer-badge {{ padding: 5px 12px; border-radius: 20px; font-size: 0.85em; }}
        .wer-improved {{ background: #e8f5e9; color: #2e7d32; }}
        .wer-degraded {{ background: #ffebee; color: #c62828; }}
        .wer-neutral {{ background: #f5f5f5; color: #666; }}
        .wer-skipped {{ background: #e3f2fd; color: #1565c0; }}
        .text-section {{ margin: 10px 0; }}
        .text-label {{ font-weight: 600; color: #333; margin-bottom: 5px; display: flex; align-items: center; gap: 8px; }}
        .text-content {{ background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.95em; line-height: 1.6; white-space: pre-wrap; }}
        .version-badge {{ background: #4a148c; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-left: 10px; }}
        .adaptive-badge {{ background: #00695c; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-left: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>노이즈 제거 STT 비교 리포트 <span class="version-badge">V6</span> <span class="adaptive-badge">초보수적 스킵</span></h1>
        <div class="subtitle">생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | SNR ≥ {config.SNR_SKIP_THRESHOLD}dB → 노이즈 제거 스킵</div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(valid_comparisons)}</div>
            <div class="stat-label">총 청크 (유효 GT)</div>
        </div>
        <div class="stat-card skip-card">
            <div class="stat-value skip-value">{total_skipped}/{total_chunks}</div>
            <div class="stat-label">스킵된 청크 ({skip_rate:.1f}%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_baseline_wer:.1f}%</div>
            <div class="stat-label">Baseline WER</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_denoised_wer:.1f}%</div>
            <div class="stat-label">Denoised WER</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: {'#2e7d32' if avg_improvement > 0 else '#c62828' if avg_improvement < 0 else '#666'}">{avg_improvement:+.1f}%p</div>
            <div class="stat-label">WER 개선</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_baseline_cer:.1f}%</div>
            <div class="stat-label">Baseline CER</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_denoised_cer:.1f}%</div>
            <div class="stat-label">Denoised CER</div>
        </div>
    </div>

    <h2>청크별 비교 (V6 - 초보수적 스킵)</h2>
"""

        for comp in comparisons:
            method = comp.get("denoise_method", "unknown")
            noise_type = comp.get("noise_type", "unknown")
            snr = comp.get("snr_db", 0)
            skipped = comp.get("skipped", False)
            skip_reason = comp.get("skip_reason", "")

            baseline_wer = comp.get("baseline_wer")
            denoised_wer = comp.get("denoised_wer")
            baseline_cer = comp.get("baseline_cer")
            denoised_cer = comp.get("denoised_cer")
            wer_imp = comp.get("wer_improvement")

            # None 값 안전하게 처리
            baseline_wer_safe = baseline_wer if baseline_wer is not None else 0
            denoised_wer_safe = denoised_wer if denoised_wer is not None else 0
            baseline_cer_safe = baseline_cer if baseline_cer is not None else 0
            denoised_cer_safe = denoised_cer if denoised_cer is not None else 0

            if skipped:
                wer_class = "wer-skipped"
                wer_display = f"WER: {baseline_wer_safe:.1f}% (스킵됨)"
                imp_display = "스킵"
                baseline_wer_str = f"{baseline_wer_safe:.1f}%"
                denoised_wer_str = f"{denoised_wer_safe:.1f}% (=원본)"
                baseline_cer_str = f"{baseline_cer_safe:.1f}%"
                denoised_cer_str = f"{denoised_cer_safe:.1f}%"
            elif baseline_wer is not None and denoised_wer is not None:
                wer_display = f"WER: {baseline_wer_safe:.1f}%→{denoised_wer_safe:.1f}%"
                baseline_wer_str = f"{baseline_wer_safe:.1f}%"
                denoised_wer_str = f"{denoised_wer_safe:.1f}%"
                baseline_cer_str = f"{baseline_cer_safe:.1f}%"
                denoised_cer_str = f"{denoised_cer_safe:.1f}%"
                if wer_imp and wer_imp > 0.5:
                    wer_class = "wer-improved"
                    imp_display = f"+{wer_imp:.1f}%p"
                elif wer_imp and wer_imp < -0.5:
                    wer_class = "wer-degraded"
                    imp_display = f"{wer_imp:.1f}%p"
                else:
                    wer_class = "wer-neutral"
                    imp_display = f"{wer_imp:.1f}%p" if wer_imp is not None else "N/A"
            else:
                wer_display = "WER: N/A (GT 없음)"
                wer_class = "wer-neutral"
                imp_display = "N/A"
                baseline_wer_str = "N/A"
                denoised_wer_str = "N/A"
                baseline_cer_str = "N/A"
                denoised_cer_str = "N/A"

            chunk_gt = comp.get("chunk_ground_truth", "") or ""
            gt_preview = chunk_gt[:200] + "..." if len(chunk_gt) > 200 else chunk_gt
            baseline_text = comp.get("baseline_text", "") or ""
            denoised_text = comp.get("denoised_text", "") or ""
            baseline_preview = baseline_text[:200]
            denoised_preview = denoised_text[:200]

            # 스킵 배지
            skip_badge = '<span class="skip-badge">⏭️ SKIPPED</span>' if skipped else ''
            method_badge = f'<span class="method-badge">{method}</span>' if not skipped else ''

            html += f"""
        <div class="chunk-card">
            <div class="chunk-header">
                <div>
                    <strong>{comp['filename']}</strong> - 청크 {comp['idx']} | 
                    {comp['start_sec']:.0f}초 - {comp['end_sec']:.0f}초
                </div>
                <div class="metrics-row">
                    {skip_badge}
                    {method_badge}
                    <span class="method-badge">{noise_type}</span>
                    <span class="snr-badge">SNR: {snr:.1f}dB</span>
                    <span class="wer-badge {wer_class}">{wer_display} ({imp_display})</span>
                </div>
            </div>

            <div class="text-section">
                <div class="text-label">
                    📝 청크 GT ({len(chunk_gt.split()) if chunk_gt else 0}단어, {len(chunk_gt)}자)
                </div>
                <div class="text-content">{gt_preview if gt_preview else '<em>GT 없음</em>'}</div>
            </div>

            <div class="text-section">
                <div class="text-label">
                    🔴 Baseline (WER: {baseline_wer_str} | CER: {baseline_cer_str})
                </div>
                <div class="text-content">{baseline_preview if baseline_preview else '<em>인식 결과 없음</em>'}</div>
            </div>

            <div class="text-section">
                <div class="text-label">
                    {'🟢' if not skipped else '⏭️'} {'Denoised' if not skipped else 'Skipped'} [{method}] (WER: {denoised_wer_str} | CER: {denoised_cer_str})
                </div>
                <div class="text-content">{denoised_preview if denoised_preview else '<em>인식 결과 없음</em>'}</div>
            </div>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML 리포트: {output_path}")

    @staticmethod
    def generate_csv(comparisons: List[dict], output_path: str):
        """CSV 저장"""
        df = pd.DataFrame(comparisons)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"CSV 데이터: {output_path}")

    @staticmethod
    def generate_statistics(results: List[dict], comparisons: List[dict],
                            output_path: str, denoiser_stats: Dict = None) -> dict:
        """통계 JSON 생성"""
        valid_results = [r for r in results if r.get("baseline_wer") is not None]
        valid_comparisons = [c for c in comparisons if c.get("baseline_wer") is not None]

        if not valid_results:
            return {}

        baseline_wers = [r["baseline_wer"] for r in valid_results]
        denoised_wers = [r["denoised_wer"] for r in valid_results]
        improvements = [r["wer_improvement"] for r in valid_results]

        # 청크별 통계
        chunk_baseline_wers = [c["baseline_wer"] for c in valid_comparisons]
        chunk_denoised_wers = [c["denoised_wer"] for c in valid_comparisons]
        chunk_improvements = [c["wer_improvement"] for c in valid_comparisons if c["wer_improvement"] is not None]

        # 스킵 통계
        skipped_chunks = [c for c in comparisons if c.get("skipped", False)]
        processed_chunks = [c for c in comparisons if not c.get("skipped", False)]

        # 스킵된 청크 vs 처리된 청크 성능 비교
        skipped_valid = [c for c in skipped_chunks if c.get("baseline_wer") is not None]
        processed_valid = [c for c in processed_chunks if c.get("baseline_wer") is not None]

        stats = {
            "version": "V6 - 초보수적 SNR 기반 스킵",
            "snr_skip_threshold": config.SNR_SKIP_THRESHOLD,
            "generated_at": datetime.now().isoformat(),
            "file_level": {
                "n_files": len(valid_results),
                "baseline_wer_mean": float(np.mean(baseline_wers)),
                "baseline_wer_std": float(np.std(baseline_wers)),
                "denoised_wer_mean": float(np.mean(denoised_wers)),
                "denoised_wer_std": float(np.std(denoised_wers)),
                "mean_improvement": float(np.mean(improvements)),
                "improved_files": sum(1 for i in improvements if i > 0),
            },
            "chunk_level": {
                "n_chunks": len(valid_comparisons),
                "baseline_wer_mean": float(np.mean(chunk_baseline_wers)) if chunk_baseline_wers else 0,
                "denoised_wer_mean": float(np.mean(chunk_denoised_wers)) if chunk_denoised_wers else 0,
                "mean_improvement": float(np.mean(chunk_improvements)) if chunk_improvements else 0,
                "improved_chunks": sum(1 for i in chunk_improvements if i > 0),
            },
            "skip_statistics": {
                "total_chunks": len(comparisons),
                "skipped_chunks": len(skipped_chunks),
                "processed_chunks": len(processed_chunks),
                "skip_rate": len(skipped_chunks) / len(comparisons) * 100 if comparisons else 0,
                "skipped_avg_snr": float(np.mean([c["snr_db"] for c in skipped_chunks])) if skipped_chunks else 0,
                "processed_avg_snr": float(np.mean([c["snr_db"] for c in processed_chunks])) if processed_chunks else 0,
            },
            "skip_vs_process_comparison": {
                "skipped_baseline_wer": float(
                    np.mean([c["baseline_wer"] for c in skipped_valid])) if skipped_valid else None,
                "skipped_denoised_wer": float(
                    np.mean([c["denoised_wer"] for c in skipped_valid])) if skipped_valid else None,
                "processed_baseline_wer": float(
                    np.mean([c["baseline_wer"] for c in processed_valid])) if processed_valid else None,
                "processed_denoised_wer": float(
                    np.mean([c["denoised_wer"] for c in processed_valid])) if processed_valid else None,
                "processed_improvement": float(np.mean([c["wer_improvement"] for c in processed_valid if c[
                    "wer_improvement"] is not None])) if processed_valid else None,
            }
        }

        # 통계 검정
        if len(baseline_wers) >= 3:
            try:
                t_stat, t_pval = ttest_rel(baseline_wers, denoised_wers)
                stats["statistical_tests"] = {
                    "paired_t_statistic": float(t_stat),
                    "paired_t_p_value": float(t_pval),
                    "significant_at_0.05": bool(t_pval < 0.05)
                }
            except Exception:
                pass

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"통계: {output_path}")
        return stats
