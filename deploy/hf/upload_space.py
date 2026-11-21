"""Create/update the Kopernik Hugging Face Space.

Requires `hf auth login` (or HF_TOKEN) first.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

SPACE = Path(__file__).resolve().parent / "space"


def main():
    if not SPACE.exists():
        raise SystemExit("Run build_space.py first")

    api = HfApi()
    who = api.whoami()["name"]
    repo_id = f"{who}/kopernik"

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(SPACE),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy Kopernik API",
    )
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"API:   https://{who.lower()}-kopernik.hf.space")


if __name__ == "__main__":
    main()
