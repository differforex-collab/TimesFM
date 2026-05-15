FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model ตอน build เลย ไม่ต้องรอตอน runtime
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers', \
trust_remote_code=True, local_files_only=False)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
