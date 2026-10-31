from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

from flask import send_file
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

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://127.0.0.1:5501",
                "http://localhost:5501",
                "http://127.0.0.1:5500",
                "http://localhost:5500",
            ]
        }
    },
)


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


def load_model():
    global model_package, model, FEATURES, TARGET, MODEL_NAME

    if not MODEL_FILE.exists():
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
    MODEL_NAME = package.get("model_name", "Saved model")

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
    Return per-feature SHAP values for one assessment.

    Positive contributions push the model probability upward;
    negative contributions push it downward relative to the
    SHAP baseline. These are model explanations, not causal effects.
    """
    initialize_shap()

    values = input_df[FEATURES].to_numpy(dtype=float)

    shap_values = shap_explainer.shap_values(
        values,
        nsamples=100,
        silent=True
    )

    shap_array = np.asarray(shap_values, dtype=float)

    # KernelExplainer normally returns (samples, features) here.
    # Handle possible extra output dimensions defensively.
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    if shap_array.ndim == 1:
        feature_contributions = shap_array
    elif shap_array.ndim == 2 and shap_array.shape[0] == 1:
        feature_contributions = shap_array[0]
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_array.shape}"
        )

    if len(feature_contributions) != len(FEATURES):
        raise ValueError("SHAP values do not match the model features.")

    base_value = float(np.asarray(
        shap_explainer.expected_value
    ).reshape(-1)[0])

    items = []

    for feature, value, contribution in zip(
        FEATURES, values[0], feature_contributions
    ):
        numeric_value = float(value)

        items.append({
            "feature": feature,
            "input_value": (
                numeric_value if np.isfinite(numeric_value) else None
            ),
            "shap_value": round(float(contribution), 6),
            "direction": (
                "higher"
                if contribution > 1e-8
                else "lower"
                if contribution < -1e-8
                else "neutral"
            ),
        })

    # Most influential features for this individual observation first.
    items.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    return {
        "available": True,
        "method": "SHAP KernelExplainer",
        "baseline_output": round(base_value, 6),
        "explained_output": round(
            float(model.predict_proba(input_df)[0][
                list(model.classes_).index(1)
            ]),
            6
        ),
        "features": items,
        "notice": (
            "SHAP values explain this model's output relative to its "
            "baseline. They are not causal effects or a clinically "
            "validated estimate of personal heart-attack risk."
        ),
    }


# ==========================================================
# 4. HEALTH CHECK
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "application": "CardioShield-AgeLess",
        "status": "running",
        "model": MODEL_NAME,
        "prediction_endpoint": "/predict",
        "message": "The prediction API is running.",
    })


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

        response = {
            "success": True,
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
            "individual_explanation": individual_explanation,
            "notice": (
                "Educational prototype only. This score comes from a "
                "fixed historical dataset, is not calibrated, and is "
                "not a real-time measurement, medical diagnosis, or "
                "personal heart-attack risk estimate."
            ),
        }

        return jsonify(response), 200

    except Exception as error:
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": "Prediction failed. Check the Flask terminal.",
            "details": str(error),
        }), 500

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
