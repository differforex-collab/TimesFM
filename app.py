from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel # เปลี่ยนมาใช้ตัวนี้แทน
import torch
import os

app = FastAPI()
model = None
MODEL_ID = "google/timesfm-2.5-200m-transformers"

@app.on_event("startup")
async def load_model():
    global model
    try:
        print(f"⏳ Loading {MODEL_ID}...")
        # ใช้ AutoModel ร่วมกับ trust_remote_code=True
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        print("✅ TimesFM 2.5 Ready!")
    except Exception as e:
        print(f"❌ Load Error: {e}")

class PriceInput(BaseModel):
    prices: list[float]

@app.post("/predict")
async def predict(body: PriceInput):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # ส่งค่า Raw Data เข้าไปได้เลย เพราะโมเดลมี RevIN ในตัว
        inputs = torch.tensor([body.prices])
        
        # พยากรณ์ไปข้างหน้า 12-13 แท่งตามที่ตั้งค่า
        with torch.no_grad():
            outputs = model.predict(inputs, horizon_len=12)
            # ดึงเฉพาะค่าพยากรณ์ออกมา
            forecast = outputs.point_forecast.tolist()[0]

        return {"forecast": forecast}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
