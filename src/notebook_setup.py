"""노트북 초기 세팅 — 프로젝트 루트 고정, .env 로드, API 키 상태 확인을 함수 하나로 묶는다.

노트북 첫 셀에서 sys.path에 프로젝트 루트를 넣어 이 모듈을 import할 수 있게 만든
다음, setup_project() 한 번으로 나머지(작업 디렉토리 고정, .env 로드, 키 인식
여부 출력)를 처리한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def find_project_root(marker: str = "src") -> Path:
    """현재 작업 디렉토리에서 위로 올라가며 marker(기본 "src") 디렉토리가 있는 지점을 찾는다."""
    root = Path.cwd()
    while not (root / marker).exists() and root != root.parent:
        root = root.parent
    return root


def setup_project(api_key_env: str = "NVIDIA_NIM_API_KEY") -> Path:
    """노트북 초반 보일러플레이트를 한 번에 처리한다.

    1) 프로젝트 루트를 찾아 작업 디렉토리와 sys.path를 고정
    2) .env 로드 — find_dotenv(usecwd=True)로 콜스택 대신 cwd 기준 탐색
       (노트북 커널의 exec 기반 실행에서 기본 find_dotenv()가 실패하는
       경우가 있어 이 방식이 더 안정적)
    3) API 키 인식 여부를 출력해 바로 눈으로 확인 가능하게 함

    반환값은 프로젝트 루트 경로.
    """
    root = find_project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    loaded_env_path = load_dotenv(find_dotenv(usecwd=True))

    print("project root:", root)
    print(".env 로드:", "성공 ✅" if loaded_env_path else "해당 없음 (.env 파일 없음)")

    if os.environ.get(api_key_env):
        print(f"{api_key_env}: 설정됨 ✅")
    else:
        print(f"{api_key_env}: 없음 ❌ — .env.example을 .env로 복사해 키를 채워 넣으세요")

    return root
