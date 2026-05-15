FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ติดตั้ง torch ก่อน
RUN pip install \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0

# ติดตั้ง package อื่น
RUN pip install \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    transformers==4.52.4 \
    accelerate==0.30.1 \
    huggingface_hub==0.32.0 \
    safetensors \
    sentencepiece \
    protobuf

# test torch
RUN python -c "import torch; print(torch.__version__)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
