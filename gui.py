import subprocess
import streamlit as st
import platform
import asyncio
import dotenv
import os
import webbrowser

system = platform.system()
dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
env = os.environ.copy()

image_path = [""]
if 'done' not in st.session_state:
    st.session_state.done = False

def pretty():
    if st.session_state.pretty_toggle:
        dotenv.set_key(dotenv_file, "PRETTY", '1')
    else:
        dotenv.set_key(dotenv_file, "PRETTY", '0')

def render_done():
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
                    placeholder.code(output, language="bash", height=256, width=1024)

                process.wait()
                asyncio.run(asyncio.sleep(3))
                placeholder.success("Setup complete!")

st.header("Done? It's time to load your models")
with st.form(key="models", clear_on_submit=False, enter_to_submit=False, width=1024):
                country, region = st.columns(2)
                with country:
                    country_model = st.text_input("Country model path", os.environ['CKPT'], width=512)
                with region:
                    regional_model = st.text_input("Regional model path", os.environ['CKPT_REG'], width=512)

                if st.form_submit_button(key="models", label="Load models", use_container_width=True, icon="🚀", icon_position="right", help="If you don't need to use a certain model, just write anything in the corresponding field"):
                    dotenv.set_key(dotenv_file, "CKPT", country_model)
                    dotenv.set_key(dotenv_file, "CKPT_REG", regional_model)
                    st.success("Model paths updated!")

st.header("Do you know the 'image' guy?")
image = st.file_uploader(label="Yeah, you do", type=["jpg", "jpeg", "png"], accept_multiple_files=False, max_upload_size=64)
if image is not None:
    image_path = [f"pics/{image.name}"]
    with open(image_path[0], "wb") as f:
        f.write(image.getbuffer())
    dotenv.set_key(dotenv_file, "INPUT_IMG", image_path[0])

pretty_col, run_col = st.columns([1, 7])
with run_col:
                if st.button("Run", key="run_button", use_container_width=True, type="primary", icon="🎯", icon_position="right"):
                    placeholder = st.empty()

                    if system == "Windows":
                            env['PYTHONIOENCODING'] = 'utf-8'
                            process = subprocess.Popen(
                                ["powershell", "chcp 65001 >$null; python torch_main.py"], shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                encoding="utf-8"
                            )
                    else:
                            env['PYTHONIOENCODING'] = 'utf-8'
                            process = subprocess.Popen(
                                ["bash", "-c", "python torch_main.py"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                encoding="utf-8"
                            )

                    output = ""
                    for line in process.stdout:
                        output += line
                            

                    st.session_state.done = True
                    process.wait()
with pretty_col:
                st.toggle(label="Pretty", value=True, key="pretty_toggle", on_change=pretty, width=256)

if st.session_state.done:
    text_output = st.empty()
    text_output.code(output, language="shellSession", height=512, width=2048)
    render_done()