"""
FastAPI backend for the California Housing price predictor.

Expected project layout (unchanged from your existing structure):

housing-predictor/
├── housing/
├── model/
│   ├── preprocessing_utils.py
│   ├── housing-pipeline.py
│   └── my_california_housing_model.pkl
├── backend/
│   ├── main.py          <- this file
│   └── requirements.txt
└── frontend/
    └── index.html

Run from the backend/ directory with:
    uvicorn main:app --reload --port 8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Make the sibling `model/` directory importable. joblib needs to be able to
# re-import `preprocessing_utils` (column_ratio, ratio_name, ClusterSimilarity)
# to unpickle the saved pipeline, since those custom objects live there.
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.append(str(MODEL_DIR))

import preprocessing_utils  # noqa: F401  (import registers the classes/functions for unpickling)

MODEL_PATH = MODEL_DIR / "my_california_housing_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find trained model at {MODEL_PATH}. "
        "Make sure my_california_housing_model.pkl is in the model/ directory."
    )

model = joblib.load(MODEL_PATH)

# Categories the pipeline's one-hot encoder was trained on. Used only to
# document/validate the API — the encoder itself handles unknown values.
OCEAN_PROXIMITY_VALUES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HouseFeatures(BaseModel):
    longitude: float = Field(..., example=-122.23)
    latitude: float = Field(..., example=37.88)
    housing_median_age: float = Field(..., example=41.0)
    total_rooms: float = Field(..., example=880.0)
    total_bedrooms: float = Field(..., example=129.0)
    population: float = Field(..., example=322.0)
    households: float = Field(..., example=126.0)
    median_income: float = Field(..., example=8.3252)
    ocean_proximity: str = Field(..., example="NEAR BAY")


class PredictionResponse(BaseModel):
    predicted_price: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="California Housing Price Predictor")

# Allow the frontend (served from a different port/origin during local dev)
# to call this API directly. Tighten allow_origins before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "ocean_proximity_options": OCEAN_PROXIMITY_VALUES}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])

    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(predicted_price=round(float(prediction), 2))