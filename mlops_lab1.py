import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

# =====================================================
# CONFIG (EDIT THIS FOR EACH EXPERIMENT)
# =====================================================
MODEL_TYPE = "rf"          # linear | lasso | ridge | rf
TEST_SPLIT = 0.3
ALPHA = 0.1               # used for lasso & ridge
N_ESTIMATORS = 100        # used for random forest
MAX_DEPTH = None          # set to int (e.g., 10) or None
USE_FEATURE_SUBSET = True # True / False
# =====================================================


# Load dataset
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

# Feature selection
if USE_FEATURE_SUBSET:
    selected_features = [
        "alcohol",
        "sulphates",
        "pH",
        "volatile acidity"
    ]
    X = df[selected_features]
else:
    X = df.drop("quality", axis=1)

y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42
)

# Model selection
if MODEL_TYPE == "linear":
    model = LinearRegression()

elif MODEL_TYPE == "lasso":
    model = Lasso(alpha=ALPHA)

elif MODEL_TYPE == "ridge":
    model = Ridge(alpha=ALPHA)

elif MODEL_TYPE == "rf":
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42
    )

else:
    raise ValueError("Invalid MODEL_TYPE selected")

# Train model
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save outputs
os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")

with open("output/results.json", "w") as f:
    json.dump(
        {
            "model": MODEL_TYPE,
            "test_split": TEST_SPLIT,
            "alpha": ALPHA,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "feature_subset": USE_FEATURE_SUBSET,
            "mse": mse,
            "r2": r2
        },
        f,
        indent=4
    )
