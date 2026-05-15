import os
os.environ["USE_TORCH"] = "1"

import traceback
import torch

# ---------------------------------------------------------
# 🔥 ไม้ตาย: บังคับให้ Transformers ยอมรับว่ามี PyTorch อยู่จริงๆ
import transformers.utils.import_utils
transformers.utils.import_utils._torch_available = True
transformers.utils.import_utils._torch_version = torch.__version__
# ---------------------------------------------------------

from transformers import TimesFm2_5ModelForPrediction
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

model = None
load_error = None

@app.on_event("startup")
async def load_model():
    global model, load_error
    try:
        print("⏳ Loading TimesFM 2.5...")
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        print("✅ MODEL READY")
    except Exception as e:
        load_error = str(e)
        print(f"❌ LOAD ERROR: {e}")
        traceback.print_exc()

@app.get("/")
def root():
    import transformers
    from transformers.utils import is_torch_available
    
    return {
        "status": "online" if model else "offline",
        "model": "google/timesfm-2.5-200m-transformers",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_available_forced": is_torch_available(),
        "error": load_error
    }

class PriceInput(BaseModel):
    prices: list[float]

@app.post("/predict")
async def predict(body: PriceInput):
    global model

    if model is None:
        return {
            "error": f"Model not loaded: {load_error}"
        }

    try:
        # สมมติฐาน: โมเดลรับ Input คล้ายๆ ของเดิม แต่อาจต้องใช้ kwargs เฉพาะ
        context = torch.tensor(
            body.prices,
            dtype=torch.float32
        ).unsqueeze(0) # อาจจะต้องเช็ค Shape อีกทีว่าโมเดลต้องการ [batch, sequence_length] หรือเปล่า

        with torch.no_grad():
            # 🔴 จุดที่เปลี่ยน: ใช้โมเดลเหมือนฟังก์ชันแทนที่จะใช้ .generate()
            outputs = model(context) 
            
            # ดึงค่า forecast ออกมาจาก outputs 
            # *หมายเหตุ: ตรงนี้อาจต้อง .squeeze() หรือดึงจาก outputs.predictions 
            # ขึ้นอยู่กับโครงสร้าง Output ของโมเดล TimesFM 2.5
            if hasattr(outputs, 'predictions'):
                forecast = outputs.predictions
            else:
                 forecast = outputs # ถ้า return ออกมาเป็น Tensor เลย
                 
            # ดึง 12 แท่งสุดท้าย
            result = forecast[0][-12:].tolist() 

        return {
            "forecast": result
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
