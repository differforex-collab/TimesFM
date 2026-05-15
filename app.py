from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel
import torch
import uvicorn
import os

app = FastAPI()

model = None

MODEL_ID = "google/timesfm-2.5-200m-transformers"

@app.on_event("startup")
async def load_model():

    global model

    print(f"⏳ Loading {MODEL_ID}...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )

    model.eval()

    print("✅ TimesFM Ready")

class PriceInput(BaseModel):
    prices: list[float]

@app.post("/predict")
async def predict(body: PriceInput):

    try:

        if len(body.prices) < 64:
            return {
                "error": "Need at least 64 candles"
            }

        inputs = torch.tensor(
            [body.prices],
            dtype=torch.float32
        )

        with torch.no_grad():

            outputs = model.predict(
                inputs,
                horizon_len=12
            )

        # รองรับหลาย output format
        if hasattr(outputs, "point_forecast"):
            forecast = outputs.point_forecast.tolist()[0]

        elif hasattr(outputs, "forecast"):
            forecast = outputs.forecast.tolist()[0]

        else:
            forecast = outputs.tolist()[0]

        return {
            "forecast": forecast
        }

    except Exception as e:

        return {
            "error": str(e)
        }

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8080)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
