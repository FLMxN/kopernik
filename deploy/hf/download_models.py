"""Download the Kopernik checkpoints into /app/models at Space startup."""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "flmxn/resnet50-streetview"
FILES = (
    "resnet50_streetview_imagenet1k.pth",
    "resnet50_streetview_imagenet1k_regional.pth",
)
DEST = Path("/app/models")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = DEST / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"already present: {target}", flush=True)
            continue
        print(f"downloading {filename} ...", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(DEST),
        )
        print(f"ready: {target}", flush=True)


if __name__ == "__main__":
    main()
