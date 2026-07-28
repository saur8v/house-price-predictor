"""
California Housing — Random Forest pipeline training script.

Directory assumption (per project structure):
housing-predictor/
├── housing/
│   └── housing.csv
├── model/
│   ├── preprocessing_utils.py   <- column_ratio, ratio_name, ClusterSimilarity live here
│   ├── housing_pipeline.py         <- this file
│   └── my_california_housing_model.pkl  <- output of this script
├── backend/
└── frontend/
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import randint

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import joblib

# Custom transformer/function definitions now live in their own module so that
# joblib can locate and re-import them when the saved pipeline is loaded
# elsewhere (e.g. in the FastAPI backend).
from preprocessing_utils import column_ratio, ratio_name, ClusterSimilarity


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
# Dataset lives one level up from model/, per your updated project structure.
housing = pd.read_csv("../housing/housing.csv")

print(housing.info())
print(housing.describe())


# ---------------------------------------------------------------------------
# 2. STRATIFIED TRAIN/TEST SPLIT (stratify by income category so the split
#    preserves the real-world income distribution)
# ---------------------------------------------------------------------------
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3, 4.5, 6, np.inf],
    labels=[1, 2, 3, 4, 5],
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# income_cat was only needed for the stratified split — drop it afterwards
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ---------------------------------------------------------------------------
# 3. SEPARATE FEATURES / LABELS
# ---------------------------------------------------------------------------
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# ---------------------------------------------------------------------------
# 4. BUILD THE PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------
# Ratio pipelines: compute a ratio between two numeric columns, impute
# missing values first, then standardize the resulting ratio.
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )

# Log pipeline: for heavily skewed numeric columns (rooms, population, etc.)
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler(),
)

# Categorical pipeline: impute with the most frequent category, then one-hot encode
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

# Geographic similarity: RBF similarity to k-means cluster centers of lat/long
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

# Fallback for any remaining numeric columns not explicitly listed above
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocessing = ColumnTransformer(
    [
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline,
)


# ---------------------------------------------------------------------------
# 5. QUICK MODEL COMPARISON (linear regression / decision tree / random forest)
# ---------------------------------------------------------------------------
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
lin_rmse = root_mean_squared_error(housing_labels, lin_reg.predict(housing))
print(f"Linear Regression training RMSE: {lin_rmse:.2f}")

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
tree_rmses = -cross_val_score(
    tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10
)
print("Decision Tree CV RMSE:\n", pd.Series(tree_rmses).describe())

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(
    forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10
)
print("Random Forest CV RMSE:\n", pd.Series(forest_rmses).describe())


# ---------------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING — GRID SEARCH (kept for reference/comparison)
# ---------------------------------------------------------------------------
full_pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ]
)

param_grid = [
    {"preprocessing__geo__n_clusters": [5, 8, 10], "random_forest__max_features": [4, 6, 8]},
    {"preprocessing__geo__n_clusters": [10, 15], "random_forest__max_features": [6, 8, 10]},
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid_search.fit(housing, housing_labels)
print("Grid search best params:", grid_search.best_params_)


# ---------------------------------------------------------------------------
# 7. HYPERPARAMETER TUNING — RANDOMIZED SEARCH (used for the final model)
# ---------------------------------------------------------------------------
param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=20),
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
)
rnd_search.fit(housing, housing_labels)
print("Randomized search best params:", rnd_search.best_params_)

final_model = rnd_search.best_estimator_

# Inspect feature importances for sanity-checking the model
feature_importances = final_model["random_forest"].feature_importances_
print(
    "Feature importances:\n",
    sorted(
        zip(feature_importances, final_model["preprocessing"].get_feature_names_out()),
        reverse=True,
    ),
)


# ---------------------------------------------------------------------------
# 8. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ---------------------------------------------------------------------------
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
print(f"Final test RMSE: {final_rmse:.2f}")

# 95% confidence interval on the test RMSE via bootstrap
def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse, confidence_level=0.95, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval
print(f"95% CI for test RMSE: [{rmse_lower:.2f}, {rmse_upper:.2f}]")


# ---------------------------------------------------------------------------
# 9. SAVE THE TRAINED PIPELINE FOR PRODUCTION
# ---------------------------------------------------------------------------
# Saves into model/ (this script's own directory), alongside preprocessing_utils.py,
# so the FastAPI backend can load both from a known relative location.
joblib.dump(final_model, "my_california_housing_model.pkl")
print("Saved pipeline to my_california_housing_model.pkl")