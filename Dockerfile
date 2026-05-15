FROM python:3.10

WORKDIR /app

# 1. ติดตั้ง System Dependencies (เพิ่ม libgomp1 สำหรับ PyTorch CPU)
RUN apt-get update && apt-get install -y \
    git build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. อัปเดต pip
RUN pip install --upgrade pip setuptools wheel

# 3. รวบการติดตั้งทุกอย่างไว้ในคำสั่งเดียว และใช้ --extra-index-url 
# เพื่อป้องกันไม่ให้ pip แอบไปโหลด PyTorch เวอร์ชันการ์ดจอมาทับ
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0+cpu \
    "transformers @ git+https://github.com/huggingface/transformers.git" \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    accelerate \
    huggingface_hub \
    safetensors \
    numpy

# 4. ตั้งค่าตัวแปรระบบ
ENV USE_TORCH=1

# 5. ทดสอบการติดตั้งในขั้นตอน Build
RUN python -c "import os; os.environ['USE_TORCH']='1'; import torch; import transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__)"
RUN python -c "import os; os.environ['USE_TORCH']='1'; from transformers import TimesFm2_5ModelForPrediction; print('TimesFm2_5 OK')"

# 6. ดาวน์โหลดโมเดล
RUN python -c "\
import os; os.environ['USE_TORCH']='1'; \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers')"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
