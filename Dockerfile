FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# ติดตั้ง transformers จาก main ก่อน (รองรับ timesfm2_5)
RUN pip install --no-cache-dir \
    "transformers @ git+https://github.com/huggingface/transformers.git" \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    accelerate \
    huggingface_hub \
    safetensors \
    numpy

# ติดตั้ง torch หลัง transformers เสมอ
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# verify
RUN python -c "import torch; print('torch:', torch.__version__)"
RUN python -c "from transformers import TimesFm2_5ModelForPrediction; print('TimesFm2_5 OK')"
RUN python -c "from transformers.utils import is_torch_available; print('torch_available:', is_torch_available())"

# Pre-download model
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers')"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
