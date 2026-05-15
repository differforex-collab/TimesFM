FROM python:3.10

WORKDIR /app

# 1. ติดตั้ง System Dependencies
RUN apt-get update && apt-get install -y \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. อัปเดต pip
RUN pip install --upgrade pip setuptools wheel

# 3. ติดตั้ง PyTorch เวอร์ชัน CPU
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 4. ติดตั้ง Transformers จาก GitHub และไลบรารีอื่นๆ
RUN pip install --no-cache-dir \
    "transformers @ git+https://github.com/huggingface/transformers.git" \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    accelerate \
    huggingface_hub \
    safetensors \
    numpy

# 5. ตั้งค่าตัวแปรระบบบังคับให้ Transformers มองเห็น PyTorch
ENV USE_TORCH=1

# 6. ทดสอบการติดตั้ง (เพิ่มการประกาศ ENV ก่อนเทสต์รัน)
RUN python -c "import os; os.environ['USE_TORCH']='1'; import torch; import transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__)"
RUN python -c "import os; os.environ['USE_TORCH']='1'; from transformers import TimesFm2_5ModelForPrediction; print('TimesFm2_5 OK')"

# 7. ดาวน์โหลดโมเดลมาเก็บไว้ใน Image
RUN python -c "\
import os; os.environ['USE_TORCH']='1'; \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers')"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
