"""Assemble the Hugging Face Space upload folder."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
SPACE = DEPLOY / "space"

SPACE_FILES = ("README.md", "Dockerfile", "requirements.txt", "download_models.py")


def main():
    if SPACE.exists():
        shutil.rmtree(SPACE)
    SPACE.mkdir(parents=True)

    for name in SPACE_FILES:
        shutil.copy2(DEPLOY / name, SPACE / name)

    shutil.copytree(
        ROOT / "backend",
        SPACE / "backend",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )

    print(f"Space folder ready: {SPACE}")
    print("Upload with:")
    print(f"  python {DEPLOY / 'upload_space.py'}")


if __name__ == "__main__":
    main()
