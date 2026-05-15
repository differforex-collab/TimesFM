FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง torch ก่อนแยก เพื่อให้แน่ใจว่าได้ CPU version
RUN pip install --no-cache-dir torch>=2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model ตอน build เลย
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers', \
trust_remote_code=True, local_files_only=False)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
