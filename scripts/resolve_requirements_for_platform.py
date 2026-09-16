"""대상 플랫폼 기준으로 마커를 평가해 requirements.txt를 평탄화한다.

pip download --platform은 wheel 태그만 바꿀 뿐, 환경 마커(sys_platform 등)는
이 스크립트를 실행 중인 머신 기준으로 평가되어 크로스 플랫폼 다운로드에서
잘못된 패키지 목록을 만든다. 이 스크립트는 대상 플랫폼의 마커 환경을 직접
지정해 올바른 패키지 목록을 만든다.

사용법:
    uv export --no-hashes -o requirements-full.txt
    uv run python scripts/resolve_requirements_for_platform.py \
        requirements-full.txt requirements-win.txt --target windows
"""

import argparse
from pathlib import Path

from packaging.requirements import Requirement

TARGET_ENVIRONMENTS = {
    "windows": {
        "os_name": "nt",
        "sys_platform": "win32",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "python_version": "3.14",
        "python_full_version": "3.14.0",
        "implementation_version": "3.14.0",
    },
    "linux": {
        "os_name": "posix",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "python_version": "3.14",
        "python_full_version": "3.14.0",
        "implementation_version": "3.14.0",
    },
    "macos": {
        "os_name": "posix",
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "python_version": "3.14",
        "python_full_version": "3.14.0",
        "implementation_version": "3.14.0",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--target", choices=TARGET_ENVIRONMENTS, required=True)
    args = parser.parse_args()

    env = TARGET_ENVIRONMENTS[args.target]
    kept: list[str] = []
    dropped: list[str] = []

    for raw_line in args.src.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        req = Requirement(line)
        if req.marker is None or req.marker.evaluate(env):
            req.marker = None
            kept.append(str(req))
        else:
            dropped.append(line)

    args.dst.write_text("\n".join(kept) + "\n")
    print(f"[{args.target}] kept {len(kept)}, dropped {len(dropped)}")
    for line in dropped:
        print(f"  - dropped: {line}")


if __name__ == "__main__":
    main()
