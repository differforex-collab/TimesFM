import os
os.environ["USE_TORCH"] = "1"

import traceback
import torch

# บังคับสถานะให้ Transformers ยอมรับ PyTorch
import transformers.utils.import_utils
transformers.utils.import_utils._torch_available = True
transformers.utils.import_utils._torch_version = torch.__version__

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
        # โหลดโมเดล และปรับให้อยู่ในโหมดประเมินผล (eval)
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        model = model.eval()
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
        # 1. แปลงข้อมูลลิสต์ราคาจาก EA ให้เป็น List ของ 1D Tensor ตามคู่มือ Hugging Face
        past_values = [
            torch.tensor(body.prices, dtype=torch.float32)
        ]

        # 2. ป้อนข้อมูลเข้าโมเดลโดยตรงเพื่อทำนายผล
        with torch.no_grad():
            outputs = model(past_values=past_values)

        # 3. ดึงค่าจาก mean_predictions (โมเดลจะให้เฉพาะค่าของอนาคตออกมาเลย)
        # ดึงลิสต์แรก [0] และเลือกเอาเฉพาะ 12 แท่งอนาคตแรก [:12] ตามที่ EA ต้องการ
        forecast_all = outputs.mean_predictions[0].tolist()
        result = forecast_all[:12]

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
