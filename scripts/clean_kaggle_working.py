"""
Kaggle Working 디렉토리 정리 스크립트
- pretrained_models는 보존 (재다운로드 방지)
- 나머지 결과 폴더들 자동 감지 후 삭제
"""

import shutil
import os
from pathlib import Path
from datetime import datetime

# ============================================================
# 설정
# ============================================================
WORKING_DIR = "/kaggle/working"

# 절대 삭제하면 안 되는 폴더 (화이트리스트)
PRESERVE_FOLDERS = {
    "pretrained_models",  # 모델 캐시 (다운로드 오래 걸림)
    "sepformer",          # Sepformer 모델
    ".cache",             # 시스템 캐시
}

# 삭제할 파일 확장자 (루트에 있는 파일들)
DELETE_EXTENSIONS = {
    ".wav", ".mp3", ".flac",  # 오디오
    ".csv", ".json",          # 데이터
    ".png", ".jpg", ".html",  # 결과물
    ".log",                   # 로그
}

# ============================================================
# 메인 로직
# ============================================================
def get_folder_size(path: str) -> int:
    """폴더 크기 (bytes)"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_folder_size(entry.path)
    except PermissionError:
        pass
    return total


def format_size(size_bytes: int) -> str:
    """바이트를 읽기 좋게 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def scan_working_dir():
    """작업 디렉토리 스캔"""
    if not os.path.exists(WORKING_DIR):
        print(f"❌ 디렉토리 없음: {WORKING_DIR}")
        return [], []
    
    folders_to_delete = []
    files_to_delete = []
    
    for entry in os.scandir(WORKING_DIR):
        if entry.is_dir():
            if entry.name not in PRESERVE_FOLDERS:
                size = get_folder_size(entry.path)
                folders_to_delete.append({
                    "path": entry.path,
                    "name": entry.name,
                    "size": size
                })
        elif entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in DELETE_EXTENSIONS:
                files_to_delete.append({
                    "path": entry.path,
                    "name": entry.name,
                    "size": entry.stat().st_size
                })
    
    return folders_to_delete, files_to_delete


def cleanup(dry_run: bool = False):
    """
    정리 실행
    
    Args:
        dry_run: True면 실제 삭제 안 하고 미리보기만
    """
    print("=" * 60)
    print(f"🧹 Kaggle Working 디렉토리 정리")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   모드: {'미리보기 (Dry Run)' if dry_run else '실제 삭제'}")
    print("=" * 60)
    
    folders, files = scan_working_dir()
    
    if not folders and not files:
        print("\n✨ 정리할 항목이 없습니다!")
        return
    
    total_size = 0
    
    # 폴더 목록
    if folders:
        print(f"\n📁 삭제 대상 폴더 ({len(folders)}개):")
        print("-" * 50)
        for f in sorted(folders, key=lambda x: x['size'], reverse=True):
            print(f"  {'🗑️' if not dry_run else '👀'} {f['name']:30} {format_size(f['size']):>10}")
            total_size += f['size']
    
    # 파일 목록
    if files:
        print(f"\n📄 삭제 대상 파일 ({len(files)}개):")
        print("-" * 50)
        for f in sorted(files, key=lambda x: x['size'], reverse=True)[:10]:  # 상위 10개만
            print(f"  {'🗑️' if not dry_run else '👀'} {f['name']:30} {format_size(f['size']):>10}")
            total_size += f['size']
        if len(files) > 10:
            print(f"  ... 외 {len(files) - 10}개 파일")
    
    print("-" * 50)
    print(f"📊 총 정리 용량: {format_size(total_size)}")
    
    # 보존되는 폴더 표시
    preserved = [p for p in PRESERVE_FOLDERS if os.path.exists(os.path.join(WORKING_DIR, p))]
    if preserved:
        print(f"\n🛡️ 보존되는 폴더: {', '.join(preserved)}")
    
    if dry_run:
        print("\n" + "=" * 60)
        print("💡 실제 삭제하려면: cleanup(dry_run=False)")
        print("=" * 60)
        return
    
    # 실제 삭제
    print("\n🚀 삭제 진행 중...")
    deleted_count = 0
    error_count = 0
    
    for f in folders:
        try:
            shutil.rmtree(f['path'])
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ 실패: {f['name']} - {e}")
            error_count += 1
    
    for f in files:
        try:
            os.remove(f['path'])
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ 실패: {f['name']} - {e}")
            error_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ 완료! 삭제: {deleted_count}개, 실패: {error_count}개")
    print(f"📦 확보된 용량: {format_size(total_size)}")
    print("=" * 60)


def quick_clean():
    """빠른 정리 (확인 없이 바로 삭제)"""
    cleanup(dry_run=False)


def preview():
    """미리보기만"""
    cleanup(dry_run=True)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        quick_clean()
    else:
        # 기본: 미리보기 먼저 보여주고 확인
        preview()
        
        print("\n")
        confirm = input("정말 삭제하시겠습니까? (y/N): ").strip().lower()
        if confirm == 'y':
            cleanup(dry_run=False)
        else:
            print("취소되었습니다.")
