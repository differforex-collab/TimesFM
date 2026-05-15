import os
import sys
import traceback

os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

# force torch ให้ import ก่อนทุกอย่าง
import torch
print(f"torch: {torch.__version__}")
print(f"torch path: {torch.__file__}")

# บังคับ transformers ให้เห็น torch
import importlib
import transformers.utils
transformers.utils.import_utils._torch_available = True
transformers.utils.import_utils.torch = torch

import transformers
print(f"transformers: {transformers.__version__}")
print(f"torch available: {transformers.is_torch_available()}")

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
        print(f"⏳ Loading model: {MODEL_ID}")
        from transformers import TimesFm2_5ModelForPrediction
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="cpu"
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
        "transformers": transformers.__version__,
        "torch_available": transformers.is_torch_available(),
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
        past_values = [torch.tensor(body.prices, dtype=torch.float32)]
        with torch.no_grad():
            outputs = model(past_values=past_values, return_dict=True)
        forecast = outputs.mean_predictions.tolist()
        return {"forecast": forecast[0][:12]}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
