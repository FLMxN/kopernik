if __name__ == "__main__":
    import subprocess
    subprocess.run(["powershell", "streamlit run gui.py --server.headless true"], shell=True)
