FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch с PyPI собран под CUDA 13 (нужен драйвер >= 580). Ставим сборку под CUDA 12.6 —
# работает на драйверах >= 560. Для CPU: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch==2.14.1 --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "pizza_bot"]
