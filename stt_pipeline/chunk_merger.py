"""[9] 청크 병합기

원본: Kaggle 노트북 audio.ipynb의 "[9]" 섹션을 모듈로 분리했습니다.
"""

from difflib import SequenceMatcher
from typing import Dict, List

from .config import config


class ChunkMerger:
    """오버랩 청크 병합"""

    @staticmethod
    def merge_chunks(chunks: List[Dict]) -> str:
        """청크 텍스트 병합"""
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0].get("text", "")

        merged_text = chunks[0].get("text", "")

        for i in range(1, len(chunks)):
            current_text = chunks[i].get("text", "")
            if not current_text:
                continue

            overlap_point = ChunkMerger._find_overlap(merged_text, current_text)

            if overlap_point > 0:
                merged_text += " " + current_text[overlap_point:]
            else:
                merged_text += " " + current_text

        return merged_text.strip()

    @staticmethod
    def _find_overlap(text1: str, text2: str) -> int:
        """두 텍스트 간 겹침 지점 찾기"""
        words1 = text1.split()
        words2 = text2.split()

        if len(words1) < 3 or len(words2) < 3:
            return 0

        max_overlap = min(config.MERGE_MAX_OVERLAP_WORDS, len(words1), len(words2))

        best_overlap_start = 0
        best_similarity = 0.0

        for overlap_size in range(3, max_overlap + 1):
            end_of_text1 = " ".join(words1[-overlap_size:])
            start_of_text2 = " ".join(words2[:overlap_size])

            similarity = SequenceMatcher(None, end_of_text1, start_of_text2).ratio()

            if similarity > best_similarity and similarity >= config.MERGE_SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_overlap_start = len(" ".join(words2[:overlap_size])) + 1

        return best_overlap_start
