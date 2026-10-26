from pathlib import Path
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
app = FastAPI(title="Bank Marketing Prediction API", version="1.0.0")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class Customer(BaseModel):
    age: int = Field(ge=18, le=120)
    job: str = "management"
    marital: str = "married"
    education: str = "tertiary"
    balance: float = 0
    housing: str = "no"
    loan: str = "no"
    contact: str = "cellular"
    month: str = "may"
    campaign: int = Field(default=1, ge=0)
    pdays: int = Field(default=999, ge=-1)
    previous: int = Field(default=0, ge=0)
    poutcome: str = "unknown"

@app.get("/")
def root():
    return {"service": "Bank Marketing Prediction API", "docs": "/docs"}
@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.get("/metrics")
def get_metrics():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not trained.")
    bundle = joblib.load(MODEL_PATH)
    return bundle.get("metrics", {})
@app.post("/predict")
def predict(customer: Customer):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not trained. Run python train_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    data = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    # Dataset uses "default" as an input field, commonly absent from compact forms.
    data["default"] = "no"
    X = pd.DataFrame([data]).reindex(columns=bundle["features"])
    model = bundle["model"]
    probability = float(model.predict_proba(X)[0, 1])
    return {"prediction": "yes" if probability >= 0.5 else "no", "probability": probability, "threshold": 0.5}
