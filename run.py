if __name__ == "__main__":
    import subprocess
    import predictor, gui, torch_gradcam, torch_main

    subprocess.run(["powershell", "streamlit run gui.py --server.headless true"], shell=True)
