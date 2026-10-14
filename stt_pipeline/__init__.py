"""한국어 STT 통합 파이프라인 V6 (SNR 기반 적응형 노이즈 제거)

한국어 STT 통합 파이프라인 (V6 - 초보수적 SNR 기반 스킵)
- V5 기반 + SNR ≥ 6dB에서 노이즈 제거 스킵 (초보수적)
- V5 결과: SNR 7-9dB 영역에서 부분 스킵 시 불균형 발생
- "Do No Harm" 원칙 최대화: SNR < 6dB에서만 처리
- 논문 기여: 최적 SNR 임계값 탐색 (6dB vs 8dB vs 15dB)
"""
import warnings

warnings.filterwarnings("ignore")
