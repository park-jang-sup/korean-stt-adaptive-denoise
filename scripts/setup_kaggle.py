# ==================================================================================
# Kaggle T4×2 환경 설치 스크립트 (개선 버전)
# - 버전 고정으로 재현성 보장
# - GPU/CUDA 확인
# - 설치 실패 시 재시도
# - Kaggle 기본 패키지는 스킵
# ==================================================================================

import subprocess
import sys
import os


def run_cmd(cmd, retries=2, shell=False):
    """명령 실행 (재시도 지원)"""
    for attempt in range(retries):
        try:
            if shell:
                subprocess.check_call(cmd, shell=True, stderr=subprocess.DEVNULL)
            else:
                subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as e:
            if attempt < retries - 1:
                print(f"  ⚠️ 재시도 중... ({attempt + 1}/{retries})")
            else:
                print(f"  ❌ 실패: {e}")
                return False
    return False


def pip_install(package, version=None, quiet=True):
    """pip 설치"""
    pkg_str = f"{package}=={version}" if version else package
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("--quiet")
    cmd.append(pkg_str)
    return run_cmd(cmd)


def check_import(import_name):
    """import 가능 여부 확인"""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_gpu():
    """GPU 및 CUDA 확인"""
    print("\n" + "=" * 70)
    print("🔍 GPU 환경 확인")
    print("=" * 70)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA 사용 가능")
            print(f"   PyTorch: {torch.__version__}")
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   GPU 수: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024 ** 3
                print(f"   GPU {i}: {props.name} ({mem_gb:.1f}GB)")
            return gpu_count
        else:
            print("⚠️ CUDA 사용 불가 - CPU 모드로 실행됩니다")
            return 0
    except ImportError:
        print("❌ PyTorch 없음")
        return 0


def install_core_packages():
    """핵심 패키지 설치"""
    print("\n" + "=" * 70)
    print("📦 핵심 패키지 설치")
    print("=" * 70)

    # 버전 고정 패키지 목록 (Kaggle 2025.04 기준 테스트됨)
    packages = [
        # (패키지명, 버전, import명)
        ("faster-whisper", "1.0.3", "faster_whisper"),
        ("noisereduce", "3.0.2", "noisereduce"),
        ("jiwer", "3.0.4", "jiwer"),
        ("speechbrain", "1.0.1", "speechbrain"),
    ]

    # Kaggle에 이미 있는 건 스킵
    skip_packages = {"scipy", "librosa", "soundfile", "tqdm", "matplotlib", "seaborn", "pandas", "numpy"}

    results = []
    for pkg, version, import_name in packages:
        if pkg in skip_packages:
            print(f"  ⏭️ {pkg}: Kaggle 기본 패키지 (스킵)")
            results.append((pkg, "skipped"))
            continue

        if check_import(import_name):
            print(f"  ✅ {pkg}: 이미 설치됨")
            results.append((pkg, "exists"))
        else:
            print(f"  📥 {pkg}=={version} 설치 중...")
            if pip_install(pkg, version):
                print(f"  ✅ {pkg}: 설치 완료")
                results.append((pkg, "installed"))
            else:
                print(f"  ❌ {pkg}: 설치 실패")
                results.append((pkg, "failed"))

    return results


def install_mecab():
    """Mecab 설치 (한국어 형태소 분석)"""
    print("\n" + "=" * 70)
    print("🇰🇷 Mecab 설치 (한국어 형태소 분석)")
    print("=" * 70)

    # 이미 설치되어 있는지 확인
    try:
        from konlpy.tag import Mecab
        m = Mecab()
        m.morphs("테스트")
        print("  ✅ Mecab: 이미 설치됨")
        return True
    except:
        pass

    print("  📥 apt 패키지 설치 중...")

    # apt 업데이트
    if not run_cmd("apt-get update -qq", shell=True):
        print("  ⚠️ apt update 실패 - 계속 진행")

    # mecab 설치
    apt_packages = ["mecab", "libmecab-dev", "mecab-ipadic-utf8"]
    for pkg in apt_packages:
        run_cmd(f"apt-get install -y -qq {pkg}", shell=True)

    # mecab-ko-dic (한국어 사전)
    print("  📥 한국어 사전 설치 중...")
    run_cmd(
        "bash -c 'cd /tmp && git clone --depth 1 https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git && cd Mecab-ko-for-Google-Colab && bash install_mecab-ko_on_colab_light_220429.sh'",
        shell=True)

    # Python 패키지
    print("  📥 Python 패키지 설치 중...")
    pip_install("mecab-python3")
    pip_install("konlpy")

    # 확인
    try:
        from konlpy.tag import Mecab
        m = Mecab()
        result = m.morphs("설치 테스트")
        print(f"  ✅ Mecab 설치 완료: {result}")
        return True
    except Exception as e:
        print(f"  ⚠️ Mecab 설치 실패: {e}")
        print("  💡 WER 계산은 가능하지만 형태소 분석 기반 평가는 불가")
        return False


def verify_installation():
    """설치 검증"""
    print("\n" + "=" * 70)
    print("🔬 설치 검증")
    print("=" * 70)

    checks = [
        ("faster_whisper", "Whisper STT"),
        ("noisereduce", "Noise Reduction"),
        ("speechbrain", "SpeechBrain (Sepformer)"),
        ("jiwer", "WER 계산"),
        ("librosa", "오디오 처리"),
        ("torch", "PyTorch"),
    ]

    all_ok = True
    for module, desc in checks:
        if check_import(module):
            print(f"  ✅ {desc}: OK")
        else:
            print(f"  ❌ {desc}: 실패")
            all_ok = False

    return all_ok


def main():
    print("=" * 70)
    print("🚀 Kaggle T4×2 STT 파이프라인 환경 설정")
    print("=" * 70)

    # 1. GPU 확인
    gpu_count = check_gpu()

    # 2. 핵심 패키지 설치
    results = install_core_packages()

    # 3. Mecab 설치 (선택적 - 실패해도 계속)
    mecab_ok = install_mecab()

    # 4. 검증
    all_ok = verify_installation()

    # 5. 결과 요약
    print("\n" + "=" * 70)
    print("📊 설치 결과 요약")
    print("=" * 70)

    installed = sum(1 for _, status in results if status == "installed")
    failed = sum(1 for _, status in results if status == "failed")

    print(f"  GPU: {gpu_count}개 {'✅' if gpu_count > 0 else '⚠️ CPU 모드'}")
    print(f"  패키지: 신규 설치 {installed}개, 실패 {failed}개")
    print(f"  Mecab: {'✅' if mecab_ok else '⚠️ 미설치 (WER 계산은 가능)'}")

    if all_ok:
        print("\n" + "=" * 70)
        print("✨ 모든 설치 완료! 이제 메인 코드를 실행하세요.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️ 일부 설치 실패 - 위 로그를 확인하세요.")
        print("=" * 70)


if __name__ == "__main__":
    main()
