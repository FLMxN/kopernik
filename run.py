if __name__ == "__main__":
    import subprocess, platform
    import predictor, gui, torch_gradcam, torch_main

    system = platform.system()
    match system:
        case "Windows":
            subprocess.run(["powershell", "streamlit run gui.py --server.runOnSave false --server.headless true"], shell=True)
        case "Linux" | "Darwin":
            subprocess.run(["bash", "-c", "streamlit run gui.py --server.runOnSave false --server.headless true"], shell=False)