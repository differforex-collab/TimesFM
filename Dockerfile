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

# remove transformers เก่า
RUN pip uninstall -y transformers || true

# install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    accelerate==0.30.1 \
    huggingface_hub==0.32.0 \
    safetensors \
    sentencepiece \
    protobuf \
    numpy

# IMPORTANT:
# install transformers stable branch ที่รองรับ TimesFM2.5
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git@v4.53.0

# verify
RUN python -c "import torch; print('Torch:', torch.__version__)"

RUN python -c "import transformers; \
print('Transformers:', transformers.__version__); \
print('Torch available:', transformers.is_torch_available())"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
