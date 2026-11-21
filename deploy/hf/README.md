---
title: Kopernik API
emoji: 🌍
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Geolocation research API with Grad-CAM
---

# Kopernik API

FastAPI backend for the Kopernik geolocation demo.

- Country and region ResNet-50 checkpoints are downloaded from
  [`flmxn/resnet50-streetview`](https://huggingface.co/flmxn/resnet50-streetview).
- `POST /api/inference` accepts an image and returns country/region predictions
  plus Grad-CAM images.
- `GET /health` reports whether the models are loaded.

The Space is CPU-only. The first request after a sleep can take up to a minute.
