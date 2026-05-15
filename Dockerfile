FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# CPU torch
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ลบ transformers เก่าออกก่อน
RUN pip uninstall -y transformers || true

# ติดตั้ง dependencies อื่น
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    accelerate==0.30.1 \
    huggingface_hub>=0.24.0 \
    safetensors \
    sentencepiece \
    protobuf \
    numpy

# ติดตั้ง transformers จาก main branch (รองรับ timesfm2_5)
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git@main

# verify
RUN python -c "import torch; print('Torch:', torch.__version__)"
RUN python -c "import transformers; \
print('Transformers:', transformers.__version__); \
print('Torch available:', transformers.is_torch_available())"

# Pre-download model ตอน build
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers', \
trust_remote_code=True)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
