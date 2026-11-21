<div align="center">

<img src="src/assets/kopernik.svg" alt="Kopernik logo" style="width:35%; height:auto;">

# Kopernik

<p align="center">
  <a href="https://github.com/FLMxN/kopernik/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-a6e3a1?style=for-the-badge&logo=opensourceinitiative&logoColor=cdd6f4&labelColor=181825" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=cdd6f4&labelColor=181825" alt="Python Version">
  <img src="https://img.shields.io/badge/torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=cdd6f4&labelColor=181825" alt="PyTorch">
  <img src="https://img.shields.io/badge/status-stable-green?style=for-the-badge" alt="Stable">
  <a href="https://github.com/FLMxN/kopernik/stargazers"><img src="https://img.shields.io/github/stars/FLMxN/kopernik?style=for-the-badge&logo=github&logoColor=cdd6f4&label=stars&labelColor=181825&color=f9e2af" alt="GitHub Stars"></a>
</p>

Open-source deep learning research tool for inspecting geographic data based on landscape pictures inside or outside of urbanity via neural networks.

**Live demo:** [flmxn.github.io/kopernik](https://flmxn.github.io/kopernik/)

</div>

## Table of contents
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Python dependencies](#python-dependencies)
  - [Model and input setup](#model-and-input-setup)
- [Understanding predictions](#understanding-predictions)
- [Inference](#inference)
  - [CLI](#cli)
  - [GUI](#gui)
  - [Pretty output digest](#pretty-output-digest)
- [License](#license)

## Architecture
Kopernik uses two **ResNet-50** classifiers initialized with **ImageNet-1K** weights:

- The country classifier predicts **56** country labels.
- The regional classifier predicts one of **7** project-defined regions.

The training scripts use different datasets: [torch_trainer.py](backend/torch_trainer.py) loads the [grouped country dataset](https://huggingface.co/datasets/stochastic/random_streetview_images_grouped_v1), while [torch_trainer_reg.py](backend/torch_trainer_reg.py) loads the [panorama dataset](https://huggingface.co/datasets/stochastic/random_streetview_images_pano_v0.0.2).

**Dataset credit:** [stochastic](https://huggingface.co/stochastic) (Winson Truong).

## Project layout
- [backend/](backend): Python inference, training, API, and Streamlit application.
- [desktop/](desktop): Electron launchers for the React and Streamlit apps.
- [src/](src): React/Vite frontend and its assets.
- [pics/](pics): sample and uploaded input images.
- [output/](output): generated Grad-CAM images and checked-in examples.
- Root-level `package.json`, `vite.config.js`, and `index.html`: npm and Vite configuration.

## Installation
### Requirements
| Requirement | Details |
|---|---|
| **Python** | **3.12** or newer |
| **Dependencies** | see below |
| **Node.js** and **npm** | for the Electron desktop app |
| **CUDA**-compatible **NVIDIA** GPU | Recommended for faster inference and training; CPU inference is *also supported*. |

### Python dependencies
Install PyTorch using the command recommended by the [official PyTorch installer](https://pytorch.org/get-started/locally/) for your OS and CUDA setup, then install the remaining packages:
```
python -m pip install scikit-learn datasets numpy opencv-python pillow tqdm python-dotenv streamlit
```

### Model and input setup
1. Download the fine-tuned checkpoint file(s) you want to use from the [Kopernik Hugging Face model repository](https://huggingface.co/flmxn/resnet50-streetview).
2. Copy [.env.example](.env.example) to `.env` and set the paths to your checkpoint(s) and input image:
```
CKPT=path/to/country_checkpoint.pth
CKPT_REG=path/to/regional_checkpoint.pth
INPUT_IMG=pics/us1.png
PRETTY=1
```

`CKPT` and `CKPT_REG` can point to the country and regional checkpoints respectively. The CLI can run either model independently; set `INPUT_IMG` to an image file for CLI inference. `PRETTY=1` enables the summarized output. Use `PRETTY=0` for diagnostic output. The `.env.example` file also lists `COUNTRY_IMG` and `REGION_IMG`, which are updated when visualizations are generated.

To train models instead, edit the settings in [torch_trainer.py](backend/torch_trainer.py) (country) or [torch_trainer_reg.py](backend/torch_trainer_reg.py) (regional) and run the corresponding script as a module from the repository root:
```
python -m backend.torch_trainer
python -m backend.torch_trainer_reg
```
Training downloads its dataset from Hugging Face and requires the training dependencies above, including PyTorch and torchvision.

## Understanding predictions

> [!NOTE]
> This section summarizes observed behavior of the current developer checkpoints. Predictions and label tendencies may change with different models or training data.

> [!IMPORTANT]
> It`s a common mistake to take prediction as-is, when working with a mathematical model (such as this one). Interpret labels in the context of the model and its training data; the model may respond to visual similarities rather than country-specific features.

Predictions are estimates from the model, not verified locations. Scores depend on the checkpoint and training data; treat them as relative model outputs, not calibrated measures of geographic certainty.

The country model's 56 class indices follow the checkpoint's `label_mapping` (alphabetical ISO 3166-1 alpha-2 order: `AD, AE, AR, AU, ...`). The regional model stores its 7 labels the same way. The backend reads this mapping from the checkpoint at load time, so displayed names always match the trained class order.

The labels can also act as visual "attractors": a prediction may reflect shared image features rather than the actual country. With the corrected label order, the previous observations translate to:

- **Null point attractors:** **Malaysia (MY)**, **Slovenia (SI)**, **Switzerland (CH)**, **Canada (CA)**, and **Germany (DE)** often appear for ambiguous images, such as scenes with visual noise or few distinctive details. Treat these predictions as especially unreliable.
- **Common attractors:**
  - **South Korea (KR)** and **United Arab Emirates (AE)** can represent developed, often Asian-looking landscapes, but may also appear for scenes from China, India, Japan, Australia, the USA, Russia, Europe, or the Americas.
  - **Russia (RU)** can appear for post-Soviet and other Eastern European landscapes, including rural or small urban scenes.
  - **Ukraine (UA)** can appear for Balkan and Eastern European landscapes, including small urban scenes.
  - **Hong Kong (HK)** can appear for northern European landscapes and, in some cases, large urban scenes in East Asia.
  - **Peru (PE)** can appear for European urban landscapes, including scenes from France, Portugal, and Italy, as well as some American landscapes.
- **Particular attractors:**
  - **Netherlands (NL)**, **Iceland (IS)**, and **Norway (NO)** can appear for broad, nonspecific scenes such as highways.
  - **Latvia (LV)** can appear for lush green vegetation.
  - **Japan (JP)** is recognized reliably on Japanese landscapes; the previous "Botswana" observation was a label-order artifact.
  - **Australia (AU)** can appear for neighborhoods in the USA.
  - **New Zealand (NZ)** can appear for townhouse settlements in North America or Oceania.
  - **Brazil (BR)** can appear for visually noisy or dense landscapes in Latin America.
  - **Singapore (SG)** can appear for southern recreational or resort locations.

A proper class-bias/attractor analysis still requires a labeled validation set; these observations are qualitative.

Grad-CAM highlights image areas associated with a prediction; it is a qualitative visualization, not proof of the model's reasoning.

## Inference
### CLI
With `.env` configured and at least one checkpoint available, run inference from the repository root:
```
python -m backend.torch_main
```

The script reads `INPUT_IMG`, writes available visualizations to `output/`, and prints predictions. To enable diagnostic output for a single run, pass `verbose`:
> [!TIP]
> Run `python -m backend.torch_main verbose` to enable diagnostic output for a current inference run.

### Static browser mode

The `public/models/` directory contains FP16 ONNX exports of both classifiers.
The frontend can run them locally with ONNX Runtime Web, without a backend:

```powershell
$env:VITE_INFERENCE_MODE = "onnx"
$env:VITE_ONNX_PRECISION = "fp16"
$env:VITE_BASE = "/kopernik/"
npm run build
```

In this mode the browser uses **WebGPU** when available and falls back to
single-threaded **WASM**. The first load downloads about 90 MB of models;
dynamic Grad-CAM is not available in the static build, so only the country
and region predictions are shown. Browser-side resampling and FP16 inference
can shift probabilities slightly compared with the PyTorch backend; treat the
static demo as a fast browser preview, not a bit-for-bit replacement.

### GUI
To start the Streamlit interface in your browser:
```
python -m streamlit run backend/gui.py
```

To run it in the Electron desktop window, create a project-root `venv`, activate it, and install the Python dependencies from the section above into that environment. Then install the **npm** dependencies; the Electron launcher uses the project `venv` directly:
```
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
# Install Python packages as described above, then:
npm install
npm run dev:streamlit
```

### Pretty output digest
The images and predictions below are an example for the checked-in sample image; results can vary with the checkpoint.

![Utah, USA](pics/us1.png)
> 1260 East 50 North, Springville, UT 84663, USA
```
✅ Extracted model_state_dict from checkpoint
✅ Checkpoint loaded successfully (country)
✅ Labels loaded from checkpoint: 56 classes
✅ Extracted model_state_dict from checkpoint
✅ Checkpoint loaded successfully (regional)
✅ Labels loaded from checkpoint: 7 classes
```
Inference script successfully loaded and initialized both models: country and regional.
```
Particular predictions:
    United States (US): 99.41
    Chile (CL): 0.25
    South Africa (ZA): 0.08
    Hungary (HU): 0.05
    Lithuania (LT): 0.05
    Peru (PE): 0.01
    Netherlands (NL): 0.01
```
![Country model output](output/country_us1.png)
Country model is confident and determined enough to assume **United States of America** (US) as a top prediction. Visualization implies a definite significance of <ins>**houses** and **trees** in the left of the image</ins>.
```
Regional predictions:
    Anglosphere and Central Europe: 93.21
    Scandinavia and Northern Europe: 6.79
    Resorts and Undefined Nature: 0.00
```
![Regional model output](output/region_us1.png)
Regional model is confident and determined enough to assume **Anglosphere and Central Europe** as a top prediction. Visualization implies a definite significance of <ins>**houses** in the left of the image</ins>, a <ins>**house** in the right of the image</ins> and a <ins>**hill valley** in the top of the image</ins>.

### License
MIT, see [LICENSE](LICENSE)
