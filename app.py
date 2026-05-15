import os
import traceback

os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import transformers

print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"torch available: {transformers.is_torch_available()}")

from transformers import AutoConfig, AutoModel
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
        # timesfm2_5 ต้องโหลด config ก่อน แล้วค่อยโหลด model ด้วย trust_remote_code
        config = AutoConfig.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            MODEL_ID,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        model.eval()
        print("✅ TimesFM 2.5 Ready")
    except Exception as e:
        load_error = str(e)
        print("❌ LOAD ERROR")
        print(str(e))
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
        context = torch.tensor([body.prices], dtype=torch.float32)
        freq = torch.tensor([0])
        with torch.no_grad():
            forecast = model.forecast(context=context, freq=freq)
        if isinstance(forecast, tuple):
            forecast = forecast[0]
        if hasattr(forecast, "tolist"):
            forecast = forecast.tolist()
        return {"forecast": forecast}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
