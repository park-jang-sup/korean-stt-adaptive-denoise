"""[5] 청크 매니저 (30초 청크 분할)

원본: Kaggle 노트북 audio.ipynb의 "[5]" 섹션을 모듈로 분리했습니다.
"""

import os
from typing import Dict, List

import soundfile as sf

from .config import config
from .srt_parser import SRTParser


class ChunkManager:
    """오디오 청크 관리"""

    def __init__(self, audio_path: str, srt_segments: List[Dict] = None):
        self.audio_path = audio_path
        self.audio, self.sr = sf.read(audio_path)
        if len(self.audio.shape) > 1:
            self.audio = self.audio.mean(axis=1)
        self.duration = len(self.audio) / self.sr
        self.chunks = self._create_chunks()
        self.srt_segments = srt_segments or []

        if self.srt_segments:
            self._assign_chunk_gt()

    def _create_chunks(self) -> List[Dict]:
        """오버랩 청크 생성"""
        chunks = []
        chunk_samples = int(config.CHUNK_DURATION * self.sr)
        overlap_samples = int(config.OVERLAP_SEC * self.sr)
        hop_samples = chunk_samples - overlap_samples

        start = 0
        idx = 0

        while start < len(self.audio):
            end = min(start + chunk_samples, len(self.audio))

            chunks.append({
                "idx": idx,
                "start_sample": start,
                "end_sample": end,
                "start_sec": start / self.sr,
                "end_sec": end / self.sr,
                "audio": self.audio[start:end].copy(),
                "ground_truth": ""
            })

            if end >= len(self.audio):
                break

            start += hop_samples
            idx += 1

        return chunks

    def _assign_chunk_gt(self):
        """각 청크에 SRT 기반 GT 할당"""
        for chunk in self.chunks:
            chunk_gt = SRTParser.get_chunk_gt(
                self.srt_segments,
                chunk["start_sec"],
                chunk["end_sec"]
            )
            chunk["ground_truth"] = chunk_gt

    def get_full_gt(self) -> str:
        """전체 파일 GT 반환"""
        if not self.srt_segments:
            return ""
        return ' '.join([seg['text'] for seg in self.srt_segments])

    def save_chunk(self, chunk_idx: int, suffix: str = "") -> str:
        """청크를 임시 파일로 저장"""
        chunk = self.chunks[chunk_idx]

        base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        temp_path = f"/tmp/{base_name}_chunk{chunk_idx}{suffix}.wav"

        sf.write(temp_path, chunk["audio"], self.sr)
        return temp_path
