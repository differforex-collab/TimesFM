from fastapi import FastAPI
from pydantic import BaseModel
import timesfm
import uvicorn
import os
import traceback

app = FastAPI()

tfm = None
load_error = None

try:

    print("⏳ Loading TimesFM...")

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,
            horizon_len=12,
            context_len=64,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

    print("✅ MODEL READY")

except Exception as e:

    load_error = str(e)
    print(f"❌ LOAD ERROR: {e}")
    traceback.print_exc()


class PriceInput(BaseModel):
    prices: list[float]


@app.get("/")
def root():
    return {
        "status": "online" if tfm else "offline",
        "error": load_error
    }


@app.post("/predict")
async def predict(body: PriceInput):

    if tfm is None:
        return {"error": f"Model not loaded: {load_error}"}

    try:

        forecast, _ = tfm.forecast(
            inputs=[body.prices],
            freq=[0]
        )

        return {
            "forecast": forecast.tolist()[0]
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
