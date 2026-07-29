# California Housing Price Predictor

An end-to-end machine learning project: a trained Random Forest pipeline for predicting median house values in California, served through a FastAPI backend and used via a web dashboard where a user can enter housing data and see the predicted price.

## Demo
<img width="1897" height="972" alt="Screenshot 2026-07-29 154312" src="https://github.com/user-attachments/assets/1cf512e3-2d0f-4db7-8778-53186e761f69" />
<img width="1900" height="970" alt="Screenshot 2026-07-29 154358" src="https://github.com/user-attachments/assets/2440da9f-5c7c-4b52-b292-de13a1fb3c0c" />


## Project Overview

- **Model**: Random Forest Regressor trained on the [California Housing dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), with custom feature engineering (ratio features, log transforms, and geographic similarity via KMeans clustering).
- **Backend**: FastAPI service that loads the trained pipeline and exposes a `/predict` endpoint.
- **Frontend**: A simple HTML/JS form where a user enters housing features and receives the predicted median house value.

## Project Structure

```
housing-predictor/
├── housing/
│   └── housing.csv              # raw dataset
├── model/
│   ├── preprocessing_utils.py   # custom transformer classes/functions used by the pipeline
│   ├── train.py                 # trains the pipeline and saves it as a .pkl
│   └── my_california_housing_model.pkl   # trained pipeline (not committed — see below)
├── backend/
│   ├── main.py                  # FastAPI app serving predictions
│   └── schemas.py               # request/response models
├── frontend/
│   └── index.html               # form UI that calls the backend
├── requirements.txt
└── README.md
```

## Setup

Clone the repo and set up a virtual environment:

```bash
git clone <your-repo-url>
cd housing-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training the Model

The trained model file (`my_california_housing_model.pkl`) is **not committed to this repo** since it's a generated artifact, not source code. To generate it locally:

```bash
cd model
python train.py
```

This loads `housing/housing.csv`, builds the preprocessing + Random Forest pipeline, tunes hyperparameters via `RandomizedSearchCV`, evaluates on a held-out test set, and saves the fitted pipeline as `my_california_housing_model.pkl` in the `model/` directory. Training takes roughly 30 seconds to a few minutes depending on your machine.

## Running the Backend

With the model file generated:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

## Running the Frontend

Open `frontend/index.html` directly in a browser, or serve it:

```bash
cd frontend
python -m http.server
```

Then visit `http://localhost:8000` (or whichever port is shown) and fill in the form to get a live prediction from the backend.

## Tech Stack

- **ML**: scikit-learn, pandas, numpy, joblib
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML, vanilla JavaScript

## Notes

- The pipeline includes custom transformers (`ClusterSimilarity`, ratio/log transforms) defined in `model/preprocessing_utils.py`. These must be importable wherever the pickled model is loaded (training script, backend), which is why they live in a standalone module rather than inline in a notebook.
