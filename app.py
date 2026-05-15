import os
import traceback
import torch

print(f"✅ torch version: {torch.__version__}")

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

model = None
load_error = None
MODEL_ID = "google/timesfm-2.5-200m-transformers"


@app.on_event("startup")
async def load_model():
    global model, load_error
    try:
        print(f"⏳ Loading {MODEL_ID}...")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.bfloat16
        )
        model.eval()
        print("✅ TimesFM 2.5 Ready")
    except Exception as e:
        load_error = str(e)
        print(f"❌ LOAD ERROR: {e}")
        traceback.print_exc()


@app.get("/")
def root():
    return {
        "status": "online" if model else "offline",
        "model": MODEL_ID,
        "torch": torch.__version__,
        "error": load_error
    }


class PriceInput(BaseModel):
    prices: list[float]


@app.post("/predict")
async def predict(body: PriceInput):
    if model is None:
        return {"error": f"Model not loaded: {load_error}"}
    if len(body.prices) < 64:
        return {"error": "Need at least 64 candles"}
    try:
        inputs = torch.tensor([body.prices], dtype=torch.float32)
        with torch.no_grad():
            outputs = model.predict(inputs, horizon_len=12)
        if hasattr(outputs, "point_forecast"):
            forecast = outputs.point_forecast.tolist()[0]
        elif hasattr(outputs, "forecast"):
            forecast = outputs.forecast.tolist()[0]
        else:
            forecast = outputs.tolist()[0]
        return {"forecast": forecast}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
