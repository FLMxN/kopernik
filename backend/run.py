if __name__ == "__main__":
    import subprocess
    import sys
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(root_dir / "backend" / "gui.py"),
            "--server.runOnSave",
            "false",
            "--server.headless",
            "true",
        ],
        cwd=root_dir,
        check=True,
    )