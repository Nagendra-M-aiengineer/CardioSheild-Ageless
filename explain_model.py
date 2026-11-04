
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ==========================================================
# 1. PROJECT SETTINGS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_FILE = (
    PROJECT_DIR / "datasets" / "cardioshield_preprocessed.csv"
)

MODEL_FILE = PROJECT_DIR / "model" / "cardio_model.pkl"

OUTPUT_DIR = PROJECT_DIR / "model" / "shap"

TARGET = "heart_disease"

FEATURES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "st_slope",
    "ca",
    "thal",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_BACKGROUND_ROWS = 100
MAX_EXPLANATION_ROWS = 100


# ==========================================================
# 2. LOAD DATA AND TRAINED MODEL
# ==========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    df = pd.read_csv(DATASET_FILE)

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET]).copy()

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only classes 0 and 1.")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    package = joblib.load(MODEL_FILE)

    if isinstance(package, dict):
        model = package["model"]
        model_name = package.get("model_name", "Unknown model")
    else:
        model = package
        model_name = "Loaded model"

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba()."
        )

    print("Loaded model:", model_name)
    print("Training background rows:", len(X_train))
    print("Test rows:", len(X_test))

    # Use a reproducible subset of training data as SHAP background.
    background = X_train.sample(
        n=min(MAX_BACKGROUND_ROWS, len(X_train)),
        random_state=RANDOM_STATE,
    )

    # Explain held-out records, not the training examples.
    explain_data = X_test.sample(
        n=min(MAX_EXPLANATION_ROWS, len(X_test)),
        random_state=RANDOM_STATE,
    )

    # ======================================================
    # 3. EXPLAIN CLASS-1 PROBABILITY
    # ======================================================

    # KernelExplainer works with the full sklearn pipeline,
    # including imputation, scaling, and Calibrated Logistic Regression.
    def predict_class_1(data):
        if isinstance(data, pd.DataFrame):
            input_df = data[FEATURES]
        else:
            input_df = pd.DataFrame(data, columns=FEATURES)

        return model.predict_proba(input_df)[:, 1]

    print("\nCalculating SHAP values...")
    print("KernelExplainer may take a few minutes.")

    explainer = shap.KernelExplainer(
        predict_class_1,
        background,
    )

    shap_values = explainer.shap_values(
        explain_data,
        nsamples=100,
        silent=False,
    )

    # SHAP output shape can vary across library versions.
    values = np.asarray(shap_values)

    if values.ndim == 3:
        values = values[:, :, -1]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    # Some SHAP versions return (features, rows).
    if values.shape == (len(FEATURES), len(explain_data)):
        values = values.T

    if values.shape != (len(explain_data), len(FEATURES)):
        raise ValueError(
            f"Cannot align SHAP values {values.shape} "
            f"with data {explain_data.shape}"
        )

    # ======================================================
    # 4. SAVE SHAP VALUES
    # ======================================================

    shap_df = pd.DataFrame(
        values,
        columns=FEATURES,
        index=explain_data.index,
    )

    shap_df.index.name = "dataset_row_index"

    shap_csv = OUTPUT_DIR / "shap_values.csv"
    shap_df.to_csv(shap_csv)

    # Mean absolute SHAP contribution across explained rows.
    importance = pd.DataFrame({
        "feature": FEATURES,
        "mean_absolute_shap_value": np.abs(values).mean(axis=0),
    }).sort_values(
        "mean_absolute_shap_value",
        ascending=False,
    )

    importance_file = OUTPUT_DIR / "shap_feature_importance.csv"
    importance.to_csv(importance_file, index=False)

    # ======================================================
    # 5. SAVE FEATURE IMPORTANCE BAR CHART
    # ======================================================

    plt.figure(figsize=(10, 7))

    plot_data = importance.sort_values(
        "mean_absolute_shap_value",
        ascending=True,
    )

    plt.barh(
        plot_data["feature"],
        plot_data["mean_absolute_shap_value"],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Clinical feature")
    plt.title(
        f"{model_name}: Global SHAP Feature Importance"
    )

    plt.tight_layout()

    importance_png = OUTPUT_DIR / "shap_feature_importance.png"

    plt.savefig(
        importance_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ======================================================
    # 6. SAVE SHAP SUMMARY PLOT
    # ======================================================

    shap.summary_plot(
        values,
        explain_data,
        feature_names=FEATURES,
        show=False,
    )

    plt.title(
        f"{model_name}: SHAP Summary Plot"
    )

    plt.tight_layout()

    summary_png = OUTPUT_DIR / "shap_summary.png"

    plt.savefig(
        summary_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ======================================================
    # 7. PRINT OUTPUT LOCATIONS
    # ======================================================

    print("\nSHAP analysis completed successfully.")

    print("SHAP values CSV:", shap_csv)
    print("Feature importance CSV:", importance_file)
    print("Feature importance chart:", importance_png)
    print("SHAP summary plot:", summary_png)

    print("\nTop 5 features by mean absolute SHAP value:")
    print(importance.head(5).to_string(index=False))

    print(
        "\nNote: SHAP values describe model behavior. "
        "They do not establish medical causation or "
        "clinical validity."
    )


if __name__ == "__main__":
    main()
