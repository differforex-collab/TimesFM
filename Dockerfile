FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง torch CPU version ก่อนเลย
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ติดตั้ง package อื่น
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    transformers>=4.40.0 \
    huggingface_hub>=0.24.0 \
    accelerate>=0.27.0

# ตรวจสอบ torch ติดตั้งสำเร็จ
RUN python -c "import torch; print('torch OK:', torch.__version__)"

# Pre-download model
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers', \
trust_remote_code=True, local_files_only=False)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
