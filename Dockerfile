FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ใช้ CPU-only torch จริง
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# packages หลัก
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    accelerate==0.30.1 \
    huggingface_hub==0.32.0 \
    safetensors \
    sentencepiece \
    protobuf

# ใช้ transformers ล่าสุดจาก github
RUN pip uninstall -y transformers

RUN pip install --no-cache-dir \
git+https://github.com/huggingface/transformers.git

# เช็ค torch
RUN python -c "import torch; print('Torch OK:', torch.__version__)"

# เช็ค transformers
RUN python -c "import transformers; print('Transformers OK:', transformers.__version__)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
