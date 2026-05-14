FROM python:3.10-slim

WORKDIR /app

# ลง Tools ที่จำเป็นสำหรับ Python บางตัว
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
snapshot_download(repo_id='google/timesfm-1.0-200m-pytorch', local_files_only=False)"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
