from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel  # แก้จาก AutoModelForPrediction เป็น AutoModel 
import torch
import uvicorn
import os

app = FastAPI()
model = None
MODEL_ID = "google/timesfm-2.5-200m-transformers"

@app.on_event("startup")
async def load_model():
    global model
    try:
        print(f"⏳ Loading {MODEL_ID}...")
        # ใช้ AutoModel พร้อม trust_remote_code=True เพื่อโหลดคลาสเฉพาะของ TimesFM 2.5 
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        # ตรวจสอบว่ามี GPU ไหม ถ้าไม่มีให้ใช้ CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"✅ TimesFM 2.5 Ready on {device}!")
    except Exception as e:
        print(f"❌ Load Error: {e}")

class PriceInput(BaseModel):
    prices: list[float]

@app.post("/predict")
async def predict(body: PriceInput):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # เตรียมข้อมูล Input
        inputs = torch.tensor([body.prices])
        
        with torch.no_grad():
            # เรียกใช้ฟังก์ชัน predict ของ TimesFM 2.5 
            outputs = model.predict(inputs, horizon_len=12)
            forecast = outputs.point_forecast.tolist()[0]

        return {"forecast": forecast}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
