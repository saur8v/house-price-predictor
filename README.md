# California Housing Price Predictor

An end-to-end machine learning project: a trained Random Forest pipeline for predicting median house values in California, served through a FastAPI backend and used via a web dashboard with a live map, where a user can enter housing data and see the predicted price.

## Demo

<img width="1897" height="972" alt="Screenshot 2026-07-29 154312" src="https://github.com/user-attachments/assets/1cf512e3-2d0f-4db7-8778-53186e761f69" />
<img width="1900" height="970" alt="Screenshot 2026-07-29 154358" src="https://github.com/user-attachments/assets/2440da9f-5c7c-4b52-b292-de13a1fb3c0c" />

## Results

- **Test RMSE**: $39,565 (95% CI: $37,826 – $41,459, via bootstrap)
- Trained and evaluated on 20,640 California census records (1990 census)
- Outperforms a plain linear regression baseline (~$68,600 RMSE) on the same data

## Project Overview

- **Exploratory analysis**: Investigated relationships between location, income, and housing density to inform feature engineering.
- **Model**: Random Forest Regressor trained on the [California Housing dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), with custom feature engineering — bedroom/room ratio features, log transforms for skewed columns, and geographic similarity via KMeans clustering — tuned with `RandomizedSearchCV`.
- **Backend**: FastAPI service that loads the trained pipeline and exposes a `/predict` endpoint.
- **Frontend**: An interactive dashboard where a user enters housing features and gets the predicted median house value, with the location plotted live on a map.

## Project Structure

```
housing-predictor/
├── housing/
│   └── housing.csv              # raw dataset
├── model/
│   ├── preprocessing_utils.py   # custom transformer classes/functions used by the pipeline
│   ├── housing-pipeline.py      # trains the pipeline and saves it as a .pkl
│   ├── test-reload.py           # reloads the saved pipeline and runs a sample
│   └── my_california_housing_model.pkl   # trained pipeline (not committed — see below)
├── backend/
│   ├── main.py                  # FastAPI app serving predictions
│   └── requirements.txt
├── frontend/
│   └── index.html               # dashboard UI (form + live map) that calls the backend
└── README.md
```

## Setup

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/saur8v/house-price-predictor
cd house-price-predictor
python3 -m venv housing-env
source housing-env/bin/activate      # on Windows: housing-env\Scripts\activate
pip install -r backend/requirements.txt
```

## Training the Model

The trained model file (`my_california_housing_model.pkl`) is **not committed to this repo** since it's a generated artifact, not source code. To generate it locally:

```bash
cd model
python housing-pipeline.py
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

Open `frontend/index.html` directly in a browser, or serve it on its own port (don't reuse 8000 — that's the backend):

```bash
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500` and fill in the form to get a live prediction from the backend, with the entered location plotted on the map.

If you serve the backend somewhere other than `http://localhost:8000`, update the `API_BASE` constant near the top of the `<script>` in `index.html` to match.

## Tech Stack

- **ML**: scikit-learn, pandas, numpy, scipy, joblib
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML, vanilla JavaScript, [Leaflet.js](https://leafletjs.com/) for the interactive map

## Notes

- The pipeline includes custom transformers (`ClusterSimilarity`, ratio/log transforms) defined in `model/preprocessing_utils.py`. These must be importable wherever the pickled model is loaded (training script, backend), which is why they live in a standalone module rather than inline in a notebook.
- `total_rooms`, `total_bedrooms`, `population`, and `households` are aggregate counts per census block group (not per individual house) — this is a property of the underlying 1990 census data, not the model. See the pipeline's ratio features for how this is accounted for.
- Median house value in the source dataset is capped at $500,001, which can affect error metrics at the high end.
