"""[1] 로깅 설정

원본: Kaggle 노트북 audio.ipynb의 "[1]" 섹션을 모듈로 분리했습니다.
"""

import logging
import os
from datetime import datetime


def setup_logging(output_dir: str) -> logging.Logger:
    """체계화된 로깅 설정"""
    logger = logging.getLogger("STT_Pipeline_V6")
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        f"{output_dir}/pipeline_v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
