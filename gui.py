import subprocess
import streamlit as st
import platform
import asyncio
import dotenv
import os
import webbrowser
from pathlib import Path
from PIL import ImageGrab, Image
import io
import time

system = platform.system()
dotenv_file = Path(__file__).parent / '.env'
dotenv.load_dotenv(dotenv_file, override=True)

if 'done' not in st.session_state:
    st.session_state.done = False


def render_done(image_path):
        cnt, reg = st.columns(2)
        with cnt:
            st.image(image=f'output/country_{image_path[0].split("/")[1]}', caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=True)
        with reg:
            st.image(image=f'output/region_{image_path[0].split("/")[1]}', caption=None, width="content", use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=True)

st.set_page_config(page_title="Kopernik v2", layout="wide", initial_sidebar_state="collapsed")
_, title, _ = st.columns(3)
with title:
    st.title("Kopernik 🧩", text_alignment="center")
    gh, readme = st.columns(2, width=512, vertical_alignment="center")
    with gh:
        if st.button("github/flmxn", type="tertiary", use_container_width=True, icon="🌌", icon_position="right"):
            webbrowser.open("https://github.com/FLMxN")
    with readme:
        if st.button("readme", type="tertiary", use_container_width=True, icon="🔎", icon_position="right"):
            webbrowser.open("https://github.com/FLMxN/kopernik/blob/main/README.md")

st.header("New? Run setup to install dependencies")
if st.button("Setup", use_container_width=True, icon="⚙️", icon_position="right"):
                placeholder = st.empty()

                process = subprocess.Popen(
                    ["powershell", "pip install torch torchvision scikit-learn datasets numpy pillow tqdm pathlib dotenv"], shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                output = ""
                for line in process.stdout:
                    output += line
                    placeholder.code(output, language="bash", height=512, width=2048)

                process.wait()
                asyncio.run(asyncio.sleep(3))
                placeholder.success("Setup complete!")

st.header("Done? It's time to load your models")
with st.form(key="models", clear_on_submit=False, enter_to_submit=False, width=2048):
                country, region = st.columns(2)
                with country:
                    country_model = st.text_input("Country model path", os.environ['CKPT'], width=1024)
                with region:
                    regional_model = st.text_input("Regional model path", os.environ['CKPT_REG'], width=1024)

                if st.form_submit_button(key="models", label="Load models", use_container_width=True, icon="🚀", icon_position="right", help="If you don't need to use a certain model, just write anything in the corresponding field"):
                    dotenv.set_key(dotenv_file, "CKPT", country_model)
                    dotenv.set_key(dotenv_file, "CKPT_REG", regional_model)
                    st.success("Model paths updated!")

st.header("Do you know the 'image' guy?")
local, buffer = st.columns(2)
with local:
    image = st.file_uploader(label="Yeah, you do?", type=["jpg", "jpeg", "png"], accept_multiple_files=False, max_upload_size=64)
    st.session_state.clip = False
with buffer:
      st.markdown("<p style='font-size: 14px;'>Or maybe business card?</p>", unsafe_allow_html=True)
      image_name = None
      if st.button("From Clipboard", use_container_width=True, icon="📋", icon_position="right"):
        try:
            clip_image = ImageGrab.grabclipboard()
            if clip_image is None:
                st.error("No image in clipboard!")
            else:
                buffer_data = io.BytesIO()
                clip_image.save(buffer_data, format="PNG")
                buffer_data.seek(0)

                image_name = f"clipboard_{int(time.time())}.png"
                image_path = [f"pics/{image_name}"]
                
                with open(image_path[0], "wb") as f:
                    f.write(buffer_data.getvalue())
                
                image = Image.open(image_path[0])
                st.success(f"Loaded from clipboard!")
                st.session_state.clip = True
        except Exception as e:
            st.error(f"Failed to grab clipboard: {e}")

if image is not None:
    run_stop = False
    if image_name is None:
        # File uploader path
        image_path = [f"pics/{image.name}"]
        with open(image_path[0], "wb") as f:
            f.write(image.getbuffer())
    else:
        # Clipboard path
        image_path = [f"pics/{image_name}"]
    st.session_state.image_path = image_path
    dotenv.set_key(dotenv_file, "INPUT_IMG", image_path[0])
else:
    run_stop = True

pretty_col, run_col = st.columns([1, 7])
with run_col:
                if st.button("Run", key="run_button", use_container_width=True, type="primary", icon="🎯", icon_position="right", disabled=run_stop):
                    run_stop = True
                    dotenv_file = Path(__file__).parent / '.env'
                    dotenv.load_dotenv(dotenv_file)
                    env = os.environ.copy()
                    output = ""
                    placeholder = st.empty()

                    if system == "Windows":
                            env['PYTHONIOENCODING'] = 'utf-8'
                            process = subprocess.Popen(
                                ["powershell", "chcp 65001 >$null; python torch_main.py"], shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                encoding="utf-8",
                                cwd=str(Path(__file__).parent),
                                env=env
                            )
                    else:
                            env['PYTHONIOENCODING'] = 'utf-8'
                            process = subprocess.Popen(
                                ["bash", "-c", "python torch_main.py"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                encoding="utf-8",
                                cwd=str(Path(__file__).parent),
                                env=env
                            )

                    output = ""
                    for line in process.stdout:
                        output += line
                            
                    st.session_state.done = True
                    process.wait()
with pretty_col:
    pretty = st.toggle(label="Pretty", value=True, key="pretty_toggle", width=256)
    match pretty:
        case True:
            dotenv.set_key(dotenv_file, "PRETTY", "1")
        case False:
            dotenv.set_key(dotenv_file, "PRETTY", "0")

if st.session_state.done:
    try:
        dotenv_file = Path(__file__).parent / '.env'
        dotenv.load_dotenv(dotenv_file)
        text_output = st.empty()
        text_output.code(output, language="shellSession", height=512, width=2048)
        if hasattr(st.session_state, 'image_path'):
            render_done(st.session_state.image_path)
    except Exception as e:
        pass