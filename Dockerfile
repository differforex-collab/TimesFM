FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip ก่อน
RUN pip install --upgrade pip

# ติดตั้ง torch CPU
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ติดตั้ง packages
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    transformers==4.52.4 \
    huggingface_hub>=0.24.0 \
    accelerate>=0.30.0 \
    safetensors \
    sentencepiece \
    protobuf

# เช็ค torch
RUN python -c "import torch; print('torch OK:', torch.__version__)"

# preload model
RUN python -c "from transformers import AutoModel; \
AutoModel.from_pretrained( \
'google/timesfm-2.5-200m-transformers', \
trust_remote_code=True)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
