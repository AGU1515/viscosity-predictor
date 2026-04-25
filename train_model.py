import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from viscosity_dataset import viscosity_data

MODEL_PATH = Path("viscosity_xgboost_model.joblib")
METRICS_PATH = Path("model_metrics.json")

# Save the exact descriptor names/order used in training.
DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
DESCRIPTOR_FUNCTIONS = {name: func for name, func in Descriptors._descList}


def generate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES string in dataset: {smiles}")

    features = []

    for descriptor in Descriptors._descList:
        try:
            features.append(descriptor[1](mol))
        except Exception:
            features.append(np.nan)

    return features
def main() -> None:
    print("Generating RDKit descriptors...")

    X = viscosity_data["SMILES"].apply(generate_features).tolist()
    y = viscosity_data["log η"].values

    X = pd.DataFrame(X, columns=DESCRIPTOR_NAMES)
    X = X.replace([np.inf, -np.inf], np.nan)

    feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=7,
    )

    model = XGBRegressor(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=4
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "num_descriptors": int(len(DESCRIPTOR_NAMES)),
    }

    joblib.dump(
        {
            "model": model,
            "descriptor_names": DESCRIPTOR_NAMES,
            "feature_medians": feature_medians,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved successfully!")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nModel performance on test set:")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2:  {r2:.3f}")


if __name__ == "__main__":
    main()
