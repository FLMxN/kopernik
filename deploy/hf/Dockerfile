FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY download_models.py .

RUN mkdir -p output models && chmod 777 output

ENV CKPT=/app/models/resnet50_streetview_imagenet1k.pth \
    CKPT_REG=/app/models/resnet50_streetview_imagenet1k_regional.pth \
    INPUT_IMG=pics/us1.png \
    PRETTY=1 \
    ALLOWED_ORIGINS=https://flmxn.github.io,http://localhost:5173,http://127.0.0.1:5173

EXPOSE 7860

CMD ["sh", "-c", "python download_models.py && exec uvicorn backend.server:app --host 0.0.0.0 --port 7860"]
