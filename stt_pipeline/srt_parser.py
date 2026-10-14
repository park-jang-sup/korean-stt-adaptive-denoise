"""[4] SRT 자막 파서

원본: Kaggle 노트북 audio.ipynb의 "[4]" 섹션을 모듈로 분리했습니다.
"""

import glob
import os
import re
from typing import Dict, List, Optional

from .config import config, logger


class SRTParser:
    """SRT 자막 파일 파서"""

    @staticmethod
    def parse_srt(srt_path: str) -> List[Dict]:
        """SRT 파일 파싱"""
        segments = []

        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
        content = None

        for encoding in encodings:
            try:
                with open(srt_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            logger.warning(f"SRT 파일 인코딩 실패: {srt_path}")
            return []

        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            time_match = re.search(
                r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
                lines[1]
            )

            if time_match:
                start_time = (
                        int(time_match.group(1)) * 3600 +
                        int(time_match.group(2)) * 60 +
                        int(time_match.group(3)) +
                        int(time_match.group(4)) / 1000
                )
                end_time = (
                        int(time_match.group(5)) * 3600 +
                        int(time_match.group(6)) * 60 +
                        int(time_match.group(7)) +
                        int(time_match.group(8)) / 1000
                )

                text = ' '.join(lines[2:]).strip()
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'\{[^}]+\}', '', text)

                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })

        return segments

    @staticmethod
    def get_chunk_gt(segments: List[Dict], chunk_start: float, chunk_end: float,
                     overlap_threshold: float = 0.5) -> str:
        """청크 시간 범위에 해당하는 GT 텍스트 추출"""
        matching_texts = []

        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']
            seg_duration = seg_end - seg_start

            if seg_duration <= 0:
                continue

            overlap_start = max(seg_start, chunk_start)
            overlap_end = min(seg_end, chunk_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            overlap_ratio = overlap_duration / seg_duration

            if overlap_ratio >= overlap_threshold:
                matching_texts.append(seg['text'])

        return ' '.join(matching_texts)

    @staticmethod
    def find_srt_file(audio_path: str) -> Optional[str]:
        """오디오 파일에 대응하는 SRT 파일 찾기"""
        audio_dir = os.path.dirname(audio_path)
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]

        direct_srt = os.path.join(audio_dir, f"{audio_name}.srt")
        if os.path.exists(direct_srt):
            return direct_srt

        for suffix, replacement in config.SRT_SUFFIX_MAPPINGS.items():
            if audio_name.endswith(suffix):
                srt_name = audio_name[:-len(suffix)] + replacement + ".srt"
                srt_path = os.path.join(audio_dir, srt_name)
                if os.path.exists(srt_path):
                    return srt_path

        for srt_file in glob.glob(os.path.join(audio_dir, "*.srt")):
            srt_name = os.path.splitext(os.path.basename(srt_file))[0]
            if audio_name in srt_name or srt_name in audio_name:
                return srt_file

        return None
