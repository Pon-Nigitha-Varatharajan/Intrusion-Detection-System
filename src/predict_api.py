# src/predict_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import traceback
from src.real_time_predictor import RealTimePredictor

app = FastAPI(title="IDS RealTime Predict API")

# Load predictor at startup
predictor = RealTimePredictor(models_dir="data/models")

class SampleIn(BaseModel):
    sample: Dict[str, Any]
    model_name: str = "RF"

@app.post("/predict")
def predict(payload: SampleIn):
    try:
        if payload.model_name not in predictor.available_models():
            raise HTTPException(status_code=400, detail=f"Model not available: {predictor.available_models()}")
        out = predictor.predict_single(payload.sample, model_name=payload.model_name)
        return {"status": "ok", "result": out}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run server (example)
    uvicorn.run("src.predict_api:app", host="0.0.0.0", port=8000, reload=False)