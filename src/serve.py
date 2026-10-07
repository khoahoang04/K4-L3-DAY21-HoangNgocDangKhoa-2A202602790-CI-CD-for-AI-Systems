import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from google.cloud import storage
import uvicorn

app = FastAPI(title="Income Inference API")

model = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    label: str

def download_model_from_gcs():
    bucket_name = os.getenv("ARTIFACT_BUCKET")
    blob_name = "artifacts/current/model.joblib"
    local_path = os.path.expanduser("~/models/model.joblib")
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Model downloaded from gs://{bucket_name}/{blob_name} to {local_path}")
    return local_path

@app.on_event("startup")
def startup_event():
    global model
    try:
        model_path = download_model_from_gcs()
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load model at startup: {e}")

@app.get("/healthz")
def healthz():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

@app.post("/score", response_model=PredictionResponse)
def score(payload: PredictionRequest):
    if len(payload.features) != 10:
        raise HTTPException(status_code=400, detail="Features must contain exactly 10 values")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    
    features = [payload.features]
    pred = int(model.predict(features)[0])
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    
    return PredictionResponse(prediction=pred, label=label)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)