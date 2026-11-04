
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================================
# 1. PROJECT PATHS AND SETTINGS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_FILE = (
    PROJECT_DIR / "datasets" / "cardioshield_preprocessed.csv"
)

MODEL_DIR = PROJECT_DIR / "model"

MODEL_FILE = MODEL_DIR / "cardio_model.pkl"
METRICS_FILE = MODEL_DIR / "model_comparison.csv"
REPORT_FILE = MODEL_DIR / "best_model_report.txt"

CONFUSION_MATRIX_FILE = MODEL_DIR / "confusion_matrix.png"
ROC_CURVE_FILE = MODEL_DIR / "roc_curve.png"

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


# ==========================================================
# 2. LOAD AND VALIDATE DATA
# ==========================================================

def load_data():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found:\n{DATASET_FILE}\n\n"
            "Run this command first:\n"
            "python preprocess.py"
        )

    df = pd.read_csv(DATASET_FILE)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=[TARGET]).copy()

    target_values = set(df[TARGET].unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Expected binary target values 0 and 1, "
            f"but found {target_values}"
        )

    if target_values != {0, 1}:
        raise ValueError(
            "Training data must contain both target classes: 0 and 1."
        )

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    if len(df) < 20:
        raise ValueError(
            "Too few rows to create a reliable train/test split."
        )

    print("\n========== DATASET ==========")
    print("Dataset:", DATASET_FILE)
    print("Rows:", len(df))
    print("Features:", len(FEATURES))
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y


# ==========================================================
# 3. BUILD PREPROCESSING PIPELINE
# ==========================================================

def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("clinical_features", numeric_pipeline, FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ==========================================================
# 4. DEFINE CLASSIFICATION MODELS
# ==========================================================

def build_models():
    models = {
        "Calibrated Logistic Regression": CalibratedClassifierCV(
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=5,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    except ImportError:
        print(
            "\nXGBoost is not installed and will be skipped.\n"
            "Install it with: python -m pip install xgboost"
        )

    return models


# ==========================================================
# 5. TRAIN AND EVALUATE MODELS
# ==========================================================

def evaluate_models(X_train, X_test, y_train, y_test):
    models = build_models()
    results = []
    fitted_models = {}

    for model_name, classifier in models.items():
        print(f"\nTraining: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor()),
                ("classifier", clone(classifier)),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        results.append(metrics)

        fitted_models[model_name] = {
            "pipeline": pipeline,
            "predictions": predictions,
            "probabilities": probabilities,
        }

        print(
            f"Accuracy : {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall   : {metrics['recall']:.4f}\n"
            f"F1-score : {metrics['f1_score']:.4f}\n"
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(results)

    # Preserve the existing model-selection rule.
    results_df = results_df.sort_values(
        by=["roc_auc", "f1_score"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return results_df, fitted_models


# ==========================================================
# 6. GENERATE CONFUSION MATRIX
# ==========================================================

def save_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Class 0",
            "Class 1",
        ],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "Logistic Regression - Confusion Matrix",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    fig.tight_layout()
    fig.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved confusion matrix:", CONFUSION_MATRIX_FILE)

    print("\nConfusion matrix:")
    print(cm)


# ==========================================================
# 7. GENERATE ROC CURVE
# ==========================================================

def save_roc_curve(y_test, probabilities):
    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities,
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Logistic Regression (AUC = {auc_value:.4f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Chance baseline",
    )

    ax.set_title("Logistic Regression - ROC Curve", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        ROC_CURVE_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved ROC curve:", ROC_CURVE_FILE)
    print(f"Logistic Regression ROC-AUC: {auc_value:.4f}")


# ==========================================================
# 8. SAVE MODEL, METRICS, REPORT, AND FIGURES
# ==========================================================

def save_outputs(
    results_df,
    fitted_models,
    X_train,
    X_test,
    y_train,
    y_test,
):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    best_name = "Calibrated Logistic Regression" if "Calibrated Logistic Regression" in fitted_models else results_df.iloc[0]["model"]
    best_pipeline = fitted_models[best_name]["pipeline"]

    model_package = {
        "model": best_pipeline,
        "features": FEATURES,
        "target": TARGET,
        "model_name": best_name,
        "input_format": "dataframe",
        "evaluation_metrics": results_df.iloc[0].to_dict(),
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )
    joblib.dump(
        model_package,
        MODEL_DIR / "cardioshield_model.pkl",
    )

    predictions = fitted_models[best_name]["predictions"]
    probabilities = fitted_models[best_name]["probabilities"]

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("CardioShield-AgeLess Model Evaluation\n")
        file.write("=" * 45 + "\n\n")
        file.write(f"Selected model: {best_name}\n")
        file.write(f"Target: {TARGET}\n")
        file.write(f"Input features: {', '.join(FEATURES)}\n\n")
        file.write("Comparison results:\n")
        file.write(results_df.to_string(index=False))
        file.write("\n\nBest model classification report:\n")
        file.write(report)
        file.write(
            "\n\nLimitations:\n"
            "This evaluation uses a held-out test split from the supplied "
            "dataset. It does not establish external validity, clinical "
            "calibration, or validated myocardial infarction risk.\n"
        )

    # Generate figures for the selected model.
    save_confusion_matrix(
        y_test,
        predictions,
    )

    save_roc_curve(
        y_test,
        probabilities,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\nSelected model:", best_name)
    print("Saved model:", MODEL_FILE)
    print("Saved comparison:", METRICS_FILE)
    print("Saved report:", REPORT_FILE)
    print("Saved confusion matrix:", CONFUSION_MATRIX_FILE)
    print("Saved ROC curve:", ROC_CURVE_FILE)

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)


# ==========================================================
# 9. MAIN TRAINING WORKFLOW
# ==========================================================

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n========== TRAIN / TEST SPLIT ==========")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    results_df, fitted_models = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_outputs(
        results_df,
        fitted_models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()
