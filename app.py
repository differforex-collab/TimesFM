import os
import traceback
import torch

print(f"✅ torch version: {torch.__version__}")

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModel

app = FastAPI()

model = None
load_error = None

MODEL_ID = "google/timesfm-2.5-200m-transformers"


@app.on_event("startup")
async def load_model():
    global model, load_error

    try:
        print(f"⏳ Loading {MODEL_ID}...")

        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )

        model.eval()

        print("✅ TimesFM Ready")

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
        return {"error": load_error}

    try:

        if len(body.prices) < 64:
            return {"error": "Need at least 64 candles"}

        context = torch.tensor(
            [body.prices],
            dtype=torch.float32
        )

        freq = torch.tensor([0])

        with torch.no_grad():

            forecast = model.forecast(
                context=context,
                freq=freq
            )

        if isinstance(forecast, tuple):
            forecast = forecast[0]

        return {
            "forecast": forecast.tolist()
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
