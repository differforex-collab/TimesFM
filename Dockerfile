FROM python:3.10

WORKDIR /app

# 1. ติดตั้ง System Dependencies
RUN apt-get update && apt-get install -y \
    git build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. อัปเดต pip
RUN pip install --upgrade pip setuptools wheel

# 3. ลบการล็อกเวอร์ชัน ==2.3.0 ออก เพื่อให้มันดึง PyTorch CPU เวอร์ชันล่าสุดเสมอ
# และเพิ่ม einops เข้าไป เพราะโมเดลใหม่ๆ มักจะเรียกใช้
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch \
    "transformers @ git+https://github.com/huggingface/transformers.git" \
    fastapi \
    uvicorn \
    accelerate \
    huggingface_hub \
    safetensors \
    numpy \
    einops

# 4. ตั้งค่าตัวแปรระบบ
ENV USE_TORCH=1

# 5. ทดสอบการติดตั้ง
RUN python -c "import os; os.environ['USE_TORCH']='1'; import torch; import transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__)"
RUN python -c "import os; os.environ['USE_TORCH']='1'; import transformers.utils.import_utils; transformers.utils.import_utils._torch_available = True; from transformers import TimesFm2_5ModelForPrediction; print('TimesFm2_5 OK')"

# 6. ดาวน์โหลดโมเดล
RUN python -c "\
import os; os.environ['USE_TORCH']='1'; \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='google/timesfm-2.5-200m-transformers')"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
