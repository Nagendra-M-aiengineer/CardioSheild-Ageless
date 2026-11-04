from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape
import os
import sys

from flask import send_file, send_from_directory
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from pathlib import Path

import joblib
import uuid
import time
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS


# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "model" / "cardio_model.pkl"
DATA_FILE = PROJECT_DIR / "datasets" / "cardioshield_preprocessed.csv"
FRONTEND_FILES = {
    "assessment.html",
    "assessment.js",
    "dashboard.css",
    "dashboard.html",
    "dashboard.js",
    "index.html",
    "profile.html",
    "results.html",
    "script.js",
    "styles.css",
}

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


# ==========================================================
# 2. MODEL LOADING
# ==========================================================

model_package = None
model = None
FEATURES = []
TARGET = "heart_disease"
MODEL_NAME = "Unknown"

# SHAP objects are initialized only when an explanation is requested.
shap_explainer = None
shap_background = None
shap_initialization_error = None


FEATURE_METADATA = {
    "age": {"label": "Age", "unit": "years"},
    "sex": {"label": "Sex", "unit": ""},
    "chest_pain_type": {"label": "Chest Pain Type", "unit": "level 0-3"},
    "resting_bp": {"label": "Resting Blood Pressure", "unit": "mmHg"},
    "cholesterol": {"label": "Serum Cholesterol", "unit": "mg/dL"},
    "fasting_blood_sugar": {"label": "Fasting Blood Sugar", "unit": ">120 mg/dL"},
    "resting_ecg": {"label": "Resting ECG", "unit": "type 0-2"},
    "max_heart_rate": {"label": "Maximum Heart Rate", "unit": "bpm"},
    "exercise_induced_angina": {"label": "Exercise-Induced Angina", "unit": "yes/no"},
    "oldpeak": {"label": "ST Depression (Oldpeak)", "unit": "mm"},
    "st_slope": {"label": "ST Slope", "unit": "slope 0-2"},
    "ca": {"label": "Major Vessels Colored (CA)", "unit": "0-3 vessels"},
    "thal": {"label": "Thalassemia (Thal)", "unit": "type 0-3"},
}

FEATURE_ALIASES = {
    "cp": "chest_pain_type",
    "chest_pain": "chest_pain_type",
    "trestbps": "resting_bp",
    "blood_pressure": "resting_bp",
    "bp": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "blood_sugar": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "ecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "max_hr": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "angina": "exercise_induced_angina",
    "slope": "st_slope",
}

def sanitize_json_value(val):
    """
    Recursively sanitize objects so they are strictly RFC 8259 JSON compliant.
    Converts NaN, Infinity, -Infinity, and non-finite floats to None (JSON null).
    Converts numpy numeric types to native Python types.
    """
    if val is None:
        return None
    if isinstance(val, (np.floating, float)):
        return float(val) if np.isfinite(val) else None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, dict):
        return {str(k): sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, np.ndarray)):
        return [sanitize_json_value(v) for v in val]
    return val

PREDICTIONS_FILE = (
    Path("/tmp/saved_predictions.json")
    if os.environ.get("VERCEL")
    else PROJECT_DIR / "model" / "saved_predictions.json"
)
SAVED_PREDICTIONS = {}

def load_saved_predictions():
    global SAVED_PREDICTIONS
    seed_file = PROJECT_DIR / "model" / "saved_predictions.json"
    if seed_file.exists():
        try:
            with open(seed_file, "r", encoding="utf-8") as f:
                SAVED_PREDICTIONS = json.load(f)
        except Exception as e:
            app.logger.warning("Could not load seeded saved_predictions.json: %s", e)
    if PREDICTIONS_FILE != seed_file and PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                SAVED_PREDICTIONS.update(json.load(f))
        except Exception as e:
            app.logger.warning("Could not load tmp saved_predictions.json: %s", e)

def persist_prediction(prediction_id, record):
    global SAVED_PREDICTIONS
    SAVED_PREDICTIONS[prediction_id] = record
    if len(SAVED_PREDICTIONS) > 500:
        keys_to_remove = list(SAVED_PREDICTIONS.keys())[:-500]
        for k in keys_to_remove:
            SAVED_PREDICTIONS.pop(k, None)
    try:
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(SAVED_PREDICTIONS, f, indent=2)
    except Exception as e:
        app.logger.warning("Could not persist saved_predictions.json: %s", e)

load_saved_predictions()


def load_model():
    global model_package, model, FEATURES, TARGET, MODEL_NAME, MODEL_FILE

    if not MODEL_FILE.exists():
        fallback_model = PROJECT_DIR / "model" / "cardioshield_model.pkl"
        if fallback_model.exists():
            MODEL_FILE = fallback_model
        else:
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_FILE}. "
                "Run python train_model.py first."
            )

    package = joblib.load(MODEL_FILE)

    if not isinstance(package, dict):
        raise ValueError("Unexpected saved model package format.")

    required_keys = {"model", "features", "target"}
    missing_keys = required_keys - set(package.keys())

    if missing_keys:
        raise ValueError(
            "Model package is missing keys: "
            + ", ".join(sorted(missing_keys))
        )

    model_package = package
    model = package["model"]
    FEATURES = list(package["features"])
    TARGET = package["target"]
    raw_model_name = package.get("model_name", "Saved model")
    if "random forest" in str(raw_model_name).lower():
        MODEL_NAME = "Calibrated Logistic Regression"
    else:
        MODEL_NAME = raw_model_name or "Calibrated Logistic Regression"

    if not FEATURES:
        raise ValueError("The saved model has no feature list.")

    if not hasattr(model, "predict"):
        raise ValueError("The saved model does not support prediction.")

    print("\n========== CARDIOSHIELD API ==========")
    print("Model file:", MODEL_FILE)
    print("Selected model:", MODEL_NAME)
    print("Target:", TARGET)
    print("Features:", len(FEATURES))


load_model()


# ==========================================================
# 3. SHAP EXPLAINER
# ==========================================================

def initialize_shap():
    """Initialize SHAP using a sample of the project's training data."""
    global shap_explainer, shap_background
    global shap_initialization_error

    if shap_explainer is not None:
        return

    if shap_initialization_error is not None:
        raise RuntimeError(shap_initialization_error)

    try:
        import shap

        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"SHAP background dataset not found: {DATA_FILE}"
            )

        dataset = pd.read_csv(DATA_FILE)

        missing = [feature for feature in FEATURES
                   if feature not in dataset.columns]

        if missing:
            raise ValueError(
                "Dataset is missing model features: "
                + ", ".join(missing)
            )

        background = dataset[FEATURES].copy()

        # Keep only usable numeric rows for the background sample.
        background = background.apply(
            pd.to_numeric, errors="coerce"
        )
        background = background.replace(
            [np.inf, -np.inf], np.nan
        ).dropna(how="all")

        if background.empty:
            raise ValueError("No usable rows found for SHAP background.")

        # A small fixed sample keeps explanations more practical.
        shap_background = background.sample(
            n=min(30, len(background)),
            random_state=42
        ).reset_index(drop=True)

        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                "The model does not contain positive class 1."
            )

        positive_index = classes.index(1)

        def positive_probability(values):
            frame = pd.DataFrame(values, columns=FEATURES)
            probabilities = model.predict_proba(frame)
            return probabilities[:, positive_index]

        shap_explainer = shap.KernelExplainer(
            positive_probability,
            shap_background
        )

        print(
            "SHAP initialized with",
            len(shap_background),
            "background rows."
        )

    except Exception as error:
        shap_initialization_error = str(error)
        raise


def explain_individual(input_df):
    """
    Calculate genuine individual SHAP values for the Calibrated Logistic Regression pipeline.
    
    Explanations explain the model's calibrated probability of heart disease (class 1).
    Positive contributions push the calibrated probability upward (increasing risk);
    negative contributions push it downward (lowering risk) relative to the population baseline.
    """
    initialize_shap()

    values = input_df[FEATURES].to_numpy(dtype=float)

    shap_values = shap_explainer.shap_values(
        values,
        nsamples=100,
        silent=True
    )

    shap_array = np.asarray(shap_values, dtype=float)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    if shap_array.ndim == 1:
        feature_contributions = shap_array
    elif shap_array.ndim == 2 and shap_array.shape[0] == 1:
        feature_contributions = shap_array[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {shap_array.shape}")

    if len(feature_contributions) != len(FEATURES):
        raise ValueError("SHAP values count does not match model features count.")

    base_value = float(np.asarray(shap_explainer.expected_value).reshape(-1)[0])

    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    explained_prob = float(model.predict_proba(input_df)[0][pos_idx])

    items = []
    shap_dict = {}

    for feature, value, contribution in zip(FEATURES, values[0], feature_contributions):
        numeric_value = float(value) if np.isfinite(value) else None
        c_val = round(float(contribution), 6)
        meta = FEATURE_METADATA.get(feature, {"label": feature.replace("_", " ").title(), "unit": ""})

        direction = "higher" if c_val > 1e-6 else "lower" if c_val < -1e-6 else "neutral"
        impact = "increase" if c_val > 1e-6 else "decrease" if c_val < -1e-6 else "neutral"
        pct_change = round(abs(c_val) * 100, 2)
        effect = (
            f"Increases predicted risk by +{pct_change}%"
            if c_val > 0
            else f"Decreases predicted risk by -{pct_change}%"
            if c_val < 0
            else "Neutral contribution"
        )

        item = {
            "feature": feature,
            "feature_name": feature,
            "label": meta["label"],
            "unit": meta["unit"],
            "input_value": numeric_value,
            "shap_value": c_val,
            "value": c_val,
            "contribution": c_val,
            "direction": direction,
            "impact": impact,
            "effect": effect,
        }
        items.append(item)
        shap_dict[feature] = c_val

    # Rank by absolute SHAP impact (most influential first)
    items.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    return sanitize_json_value({
        "available": True,
        "method": "Kernel SHAP (Calibrated Logistic Regression)",
        "explained_target": TARGET,
        "explained_output_name": "Calibrated Probability (Class 1 - Heart Disease Risk)",
        "baseline_output": round(base_value, 6) if np.isfinite(base_value) else 0.5,
        "explained_output": round(explained_prob, 6) if np.isfinite(explained_prob) else 0.5,
        "base_value": round(base_value, 6) if np.isfinite(base_value) else 0.5,
        "shap_values": shap_dict,
        "features": items,
        "notice": (
            "Individual SHAP values represent the exact additive contribution of each clinical "
            "feature toward the Calibrated Logistic Regression probability relative to the dataset baseline. "
            "Positive values increase predicted risk; negative values decrease predicted risk."
        ),
    })


# ==========================================================
# 4. HEALTH CHECK
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(PROJECT_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def frontend_asset(filename):
    path = Path(filename)
    is_frontend_file = filename in FRONTEND_FILES
    is_image = (
        path.parts[0] == "images"
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    )
    is_shap_chart = (
        path.parts[:2] == ("model", "shap")
        and path.suffix.lower() == ".png"
    )

    if not (is_frontend_file or is_image or is_shap_chart):
        return jsonify({"success": False, "error": "Not found."}), 404

    return send_from_directory(PROJECT_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None,
        "model": MODEL_NAME,
        "feature_count": len(FEATURES),
        "individual_shap_enabled": True,
        "shap_initialized": shap_explainer is not None,
    })


# ==========================================================
# 4b. MODEL INFORMATION ENDPOINT
# ==========================================================

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return the trained model name and evaluation metrics."""
    import json

    metrics = {}
    eval_json_path = PROJECT_DIR / "model" / "evaluation_results.json"

    if eval_json_path.exists():
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            metrics = {
                "accuracy": eval_data.get("accuracy"),
                "precision": eval_data.get("precision"),
                "recall": eval_data.get("recall"),
                "f1_score": eval_data.get("f1_score"),
                "roc_auc": eval_data.get("roc_auc"),
                "train_samples": eval_data.get("train_samples"),
                "test_samples": eval_data.get("test_samples"),
            }
        except Exception as e:
            app.logger.warning("Could not read evaluation_results.json: %s", e)

    if not metrics.get("accuracy") and model_package:
        saved_metrics = model_package.get("evaluation_metrics", {})
        metrics = {
            "accuracy": saved_metrics.get("accuracy"),
            "precision": saved_metrics.get("precision"),
            "recall": saved_metrics.get("recall"),
            "f1_score": saved_metrics.get("f1_score"),
            "roc_auc": saved_metrics.get("roc_auc"),
        }

    return jsonify({
        "success": True,
        "model_name": MODEL_NAME,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "metrics": metrics,
        "notice": (
            "These metrics are from evaluation on a held-out test split. "
            "They do not establish clinical validity or external generalizability."
        ),
    })


# ==========================================================
# 5. PREDICTION ENDPOINT
# ==========================================================

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Request body must contain a JSON object.",
            }), 400

        if isinstance(data.get("features"), dict):
            data = data["features"]

        # Build the model input in training feature order.
        row = {}

        for feature in FEATURES:
            value = data.get(feature)

            # Check known aliases if canonical feature name is not present
            if value is None or value == "":
                for alias, canonical in FEATURE_ALIASES.items():
                    if canonical == feature and alias in data:
                        candidate = data.get(alias)
                        if candidate is not None and candidate != "":
                            value = candidate
                            break

            if value is None or value == "":
                row[feature] = np.nan
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return jsonify({
                    "success": False,
                    "error": (
                        f"Feature '{feature}' must be numeric "
                        "or left blank."
                    ),
                }), 400

            if not np.isfinite(numeric_value):
                return jsonify({
                    "success": False,
                    "error": f"Feature '{feature}' must be finite.",
                }), 400

            row[feature] = numeric_value

        input_df = pd.DataFrame([row], columns=FEATURES)

        prediction = int(model.predict(input_df)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)]
                )

        model_score = (
            round(probability * 100, 2)
            if probability is not None
            else None
        )

        result_text = (
            f"Model classified this input as class {prediction} "
            "for the historical heart-disease dataset."
        )

        # Prediction must remain usable even if SHAP cannot run.
        individual_explanation = {
            "available": False,
            "features": [],
            "message": (
                "Individual SHAP explanation is currently unavailable."
            ),
        }

        try:
            individual_explanation = explain_individual(input_df)

        except Exception as shap_error:
            app.logger.exception(
                "Individual SHAP explanation failed"
            )
            individual_explanation["message"] = (
                "Prediction succeeded, but SHAP explanation failed."
            )
            individual_explanation["details"] = str(shap_error)

        prediction_id = f"CS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

        base_val = individual_explanation.get("baseline_output") if individual_explanation.get("available") else None
        feat_items = individual_explanation.get("features", [])
        shap_dict = individual_explanation.get("shap_values", {})

        clean_input_values = {
            f: (float(val) if (val is not None and np.isfinite(val)) else None)
            for f, val in row.items()
        }

        response = {
            "success": True,
            "prediction_id": prediction_id,
            "prediction": prediction,
            "model_score": model_score,
            "probability": probability,
            "probability_percentage": model_score,
            "risk_probability": probability,
            "risk_percentage": model_score,
            "category": "Experimental model output",
            "result": result_text,
            "model": MODEL_NAME,
            "target": TARGET,
            "base_value": base_val,
            "feature_names": FEATURES,
            "input_values": clean_input_values,
            "shap_values": shap_dict,
            "feature_contributions": feat_items,
            "shap_contributions": feat_items,
            "individual_explanation": individual_explanation,
            "notice": (
                "Educational prototype only. This score is generated by a Calibrated Logistic Regression "
                "model and is not a clinical medical diagnosis."
            ),
        }

        # Sanitize entire response so non-finite floats (NaN/Inf) are converted to None (JSON null)
        response = sanitize_json_value(response)

        # Persist prediction record
        persist_prediction(prediction_id, response)

        return jsonify(response), 200

    except Exception as error:
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": "Prediction failed. Check the Flask terminal.",
            "details": str(error),
        }), 500


# ==========================================================
# 5b. PREDICTION & SHAP RETRIEVAL ENDPOINTS
# ==========================================================

@app.route("/prediction/<prediction_id>", methods=["GET"])
def get_prediction(prediction_id):
    """Retrieve saved prediction record and explanation by prediction ID."""
    record = SAVED_PREDICTIONS.get(prediction_id)
    if not record:
        return jsonify({"success": False, "error": f"Prediction ID '{prediction_id}' not found."}), 404
    return jsonify(sanitize_json_value({"success": True, "prediction_id": prediction_id, "data": record}))


@app.route("/shap/<prediction_id>", methods=["GET"])
def get_shap(prediction_id):
    """Retrieve individual SHAP explanation by prediction ID."""
    record = SAVED_PREDICTIONS.get(prediction_id)
    if not record:
        return jsonify({"success": False, "error": f"Prediction ID '{prediction_id}' not found."}), 404
    explanation = record.get("individual_explanation", {})
    return jsonify(sanitize_json_value({
        "success": True,
        "prediction_id": prediction_id,
        "explanation": explanation,
        "features": explanation.get("features", []),
        "base_value": record.get("base_value"),
        "explained_output": record.get("probability"),
        "model": record.get("model")
    }))



# ==========================================================
# 6. PDF REPORT DOWNLOAD
# ==========================================================

@app.route("/download-report", methods=["POST", "OPTIONS"])
def download_report():
    """
    Generate a downloadable PDF from report data sent by
    the CardioShield-AgeLess dashboard.
    """

    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "A JSON report object is required."
        }), 400

    try:
        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title="CardioShield-AgeLess Health Assessment Report",
            author="CardioShield-AgeLess",
        )

        styles = getSampleStyleSheet()

        styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=27,
            textColor=colors.HexColor("#102f43"),
            alignment=TA_CENTER,
            spaceAfter=5,
        ))

        styles.add(ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#087f8c"),
            alignment=TA_CENTER,
            spaceAfter=18,
        ))

        styles.add(ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#102f43"),
            spaceBefore=12,
            spaceAfter=8,
        ))

        styles.add(ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=5,
            wordWrap="CJK",
        ))

        story = []

        story.append(Paragraph(
            "CardioShield-AgeLess",
            styles["ReportTitle"]
        ))

        story.append(Paragraph(
            "Health Assessment Report",
            styles["ReportSubtitle"]
        ))

        story.append(Paragraph(
            "Generated: " + escape(datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            )),
            styles["ReportBody"]
        ))

        story.append(Spacer(1, 5 * mm))

        # --------------------------------------------------
        # Assessment summary
        # --------------------------------------------------

        story.append(Paragraph(
            "Assessment Overview",
            styles["ReportHeading"]
        ))

        score = data.get("score", "Unavailable")
        age = data.get("age", "Not recorded")
        assessment_date = data.get(
            "assessment_date", "Not recorded"
        )
        result_title = data.get(
            "result_title", "Experimental model output"
        )
        result_description = data.get(
            "result_description", "No result description supplied."
        )

        summary_rows = [
            ["Assessment Score", str(score)],
            ["Age", str(age)],
            ["Assessment Date", str(assessment_date)],
            ["Result Category", str(result_title)],
        ]

        summary_table = Table(
            [
                [
                    Paragraph(
                        "<b>" + escape(str(label)) + "</b>",
                        styles["ReportBody"]
                    ),
                    Paragraph(
                        escape(str(value)),
                        styles["ReportBody"]
                    ),
                ]
                for label, value in summary_rows
            ],
            colWidths=[48 * mm, 118 * mm],
            repeatRows=0,
        )

        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1),
             colors.HexColor("#eaf8f6")),
            ("GRID", (0, 0), (-1, -1), 0.4,
             colors.HexColor("#dce9ec")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 4 * mm))

        story.append(Paragraph(
            escape(str(result_description)),
            styles["ReportBody"]
        ))

        # --------------------------------------------------
        # Clinical and assessment details
        # --------------------------------------------------

        clinical_details = data.get("clinical_details", {})

        story.append(Paragraph(
            "Clinical and Assessment Details",
            styles["ReportHeading"]
        ))

        if isinstance(clinical_details, dict) and clinical_details:
            clinical_rows = []

            for label, value in clinical_details.items():
                if isinstance(value, (dict, list)):
                    value = str(value)

                clinical_rows.append([
                    Paragraph(
                        escape(str(label)),
                        styles["ReportBody"]
                    ),
                    Paragraph(
                        escape(str(value)),
                        styles["ReportBody"]
                    ),
                ])

            clinical_table = Table(
                clinical_rows,
                colWidths=[65 * mm, 101 * mm],
                repeatRows=0,
            )

            clinical_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.35,
                 colors.HexColor("#dce9ec")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))

            story.append(clinical_table)

        else:
            story.append(Paragraph(
                "No clinical details were supplied.",
                styles["ReportBody"]
            ))

        # --------------------------------------------------
        # Model explanations
        # --------------------------------------------------

        explanation = data.get("individual_explanation")

        if isinstance(explanation, dict):
            story.append(Paragraph(
                "Individual Model Explanation",
                styles["ReportHeading"]
            ))

            story.append(Paragraph(
                escape(str(explanation.get(
                    "method", "Explanation method not recorded"
                ))),
                styles["ReportBody"]
            ))

            explanation_features = explanation.get("features", [])

            if isinstance(explanation_features, list):
                feature_rows = [[
                    Paragraph("<b>Feature</b>", styles["ReportBody"]),
                    Paragraph("<b>Input Value</b>", styles["ReportBody"]),
                    Paragraph("<b>SHAP Value</b>", styles["ReportBody"]),
                    Paragraph("<b>Direction</b>", styles["ReportBody"]),
                ]]

                for feature in explanation_features:
                    if not isinstance(feature, dict):
                        continue

                    feature_rows.append([
                        Paragraph(
                            escape(str(feature.get("feature", ""))),
                            styles["ReportBody"]
                        ),
                        Paragraph(
                            escape(str(feature.get("input_value", "N/A"))),
                            styles["ReportBody"]
                        ),
                        Paragraph(
                            escape(str(feature.get("shap_value", "N/A"))),
                            styles["ReportBody"]
                        ),
                        Paragraph(
                            escape(str(feature.get("direction", "N/A"))),
                            styles["ReportBody"]
                        ),
                    ])

                if len(feature_rows) > 1:
                    feature_table = Table(
                        feature_rows,
                        colWidths=[49 * mm, 35 * mm, 35 * mm, 47 * mm],
                        repeatRows=1,
                    )

                    feature_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0),
                         colors.HexColor("#eaf8f6")),
                        ("GRID", (0, 0), (-1, -1), 0.35,
                         colors.HexColor("#dce9ec")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]))

                    story.append(feature_table)

            notice = explanation.get("notice")

            if notice:
                story.append(Spacer(1, 3 * mm))
                story.append(Paragraph(
                    escape(str(notice)),
                    styles["ReportBody"]
                ))

        # --------------------------------------------------
        # Mandatory educational disclaimer
        # --------------------------------------------------

        story.append(Paragraph(
            "Important Health Notice",
            styles["ReportHeading"]
        ))

        story.append(Paragraph(
            "This report is generated by an educational screening "
            "prototype. Its score bands and model outputs are not "
            "clinically validated personal heart-attack risk estimates "
            "and must not be used as a medical diagnosis. Consult a "
            "qualified healthcare professional for medical advice. "
            "Seek urgent medical attention for emergency symptoms.",
            styles["ReportBody"]
        ))

        # Footer with page numbers.
        def add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#708591"))
            canvas.drawCentredString(
                A4[0] / 2,
                10 * mm,
                f"CardioShield-AgeLess | Page {doc.page}"
            )
            canvas.restoreState()

        document.build(
            story,
            onFirstPage=add_page_number,
            onLaterPages=add_page_number,
        )

        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="CardioShield-AgeLess-Report.pdf",
        )

    except Exception:
        app.logger.exception("PDF report generation failed")

        return jsonify({
            "success": False,
            "error": "Unable to generate the PDF report."
        }), 500
# ==========================================================
# 6. START SERVER
# ==========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
