
"use strict";

/*
 * CardioShield-AgeLess Dashboard
 * Displays the saved model output and individual SHAP explanations.
 * Educational prototype only; not a clinical diagnostic tool.
 */

const RESULT_KEY = "cardioResult";
const ASSESSMENT_KEY = "cardioAssessment";

function readStoredJSON(key) {
    try {
        const value = localStorage.getItem(key);
        return value ? JSON.parse(value) : null;
    } catch (error) {
        console.error("Unable to read saved data:", key, error);
        return null;
    }
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (!element) return;

    element.textContent =
        value === undefined || value === null || value === ""
            ? "--"
            : String(value);
}

/* Display the saved model output */
function updateHeartRiskDisplay(score) {
    const panel = document.getElementById("heartRiskPanel");
    const scoreElement = document.getElementById("heartRiskScore");
    const badge = document.getElementById("heartRiskBadge");
    const message = document.getElementById("heartRiskMessage");
    const category = document.getElementById("riskCategory");
    const mainScore = document.getElementById("riskPercentage");

    if (!Number.isFinite(score) || score < 0 || score > 100) {
        setText("heartRiskScore", "Unavailable");
        setText("heartRiskBadge", "Score unavailable");
        setText("heartRiskMessage", "No valid model score was returned.");
        setText("riskPercentage", "-- / 100");
        setText("riskCategory", "Result unavailable");
        return;
    }

    const color = "#0e7490";
    const light = "rgba(14, 116, 144, 0.18)";
    const label = "Experimental model output";
    const description =
        "This score comes from a fixed historical dataset. It is not " +
        "calibrated, a real-time measurement, a diagnosis, or a personal " +
        "heart-attack risk estimate.";

    if (panel) {
        panel.style.setProperty("--risk-color", color);
        panel.style.setProperty("--risk-light", light);
    }

    if (scoreElement) scoreElement.textContent = score.toFixed(1) + " / 100";
    if (badge) badge.textContent = label;
    if (message) message.textContent = description;

    if (mainScore) {
        mainScore.textContent = score.toFixed(1) + " / 100";
        mainScore.style.color = color;
    }

    if (category) {
        category.textContent = label;
        category.style.color = color;
    }
}

/* Display saved assessment details */
function loadHealthSummary(assessment) {
    if (!assessment || typeof assessment !== "object") return;

    setText("ageValue", assessment.age);
    setText("bmiValue", assessment.BMI ?? assessment.bmi);
    setText("bpValue", assessment.sysBP ?? assessment.resting_bp);
    setText(
        "cholesterolValue",
        assessment.totChol ?? assessment.cholesterol
    );
    setText(
        "glucoseValue",
        assessment.glucose ?? assessment.fasting_blood_sugar
    );

    const smoker = assessment.currentSmoker;

    if (smoker === 1 || smoker === "1" || smoker === true) {
        setText("smokingValue", "Yes");
    } else if (
        smoker === 0 || smoker === "0" || smoker === false
    ) {
        setText("smokingValue", "No");
    } else {
        setText("smokingValue", "--");
    }
}

/* Sliders do not update the model or estimate medical outcomes. */
function updateSimulation() {
    const activitySlider = document.getElementById("activitySlider");
    const smokingSlider = document.getElementById("smokingSlider");
    const dietSlider = document.getElementById("dietSlider");

    if (!activitySlider || !smokingSlider || !dietSlider) return;

    const activity = Number(activitySlider.value);
    const smoking = Number(smokingSlider.value);
    const diet = Number(dietSlider.value);

    setText("activityValue", activity + "%");
    setText("smokingReductionValue", smoking + "%");
    setText("dietValue", diet + "%");

    setText("simulationRisk", "Not calculated");
    setText(
        "simulationMessage",
        "These sliders do not change the model or estimate medical outcomes."
    );
}

/* Convert feature names to readable labels */
function formatFeatureName(name) {
    const labels = {
        chest_pain_type: "Chest pain type",
        exercise_induced_angina: "Exercise-induced angina",
        oldpeak: "Oldpeak",
        max_heart_rate: "Maximum heart rate",
        age: "Age",
        sex: "Sex",
        thal: "Thal",
        cholesterol: "Cholesterol",
        ca: "CA",
        st_slope: "ST slope",
        resting_bp: "Resting blood pressure",
        resting_ecg: "Resting ECG",
        fasting_blood_sugar: "Fasting blood sugar"
    };

    return labels[name] || name.replaceAll("_", " ");
}

/* Render one SHAP contribution safely */
function createShapRow(item) {
    const row = document.createElement("div");
    row.className = "individual-shap-row";

    const header = document.createElement("div");
    header.className = "individual-shap-header";

    const name = document.createElement("span");
    name.className = "individual-shap-name";
    name.textContent = formatFeatureName(item.feature);

    const value = Number(item.shap_value);
    const validValue = Number.isFinite(value) ? value : 0;

    const contribution = document.createElement("span");
    contribution.className = "individual-shap-value";
    contribution.textContent =
        (validValue > 0 ? "+" : "") + validValue.toFixed(5);

    contribution.style.color =
        validValue > 0 ? "#dc2626" :
        validValue < 0 ? "#2563eb" : "#64748b";

    header.append(name, contribution);

    const input = document.createElement("p");
    input.className = "individual-shap-input";

    input.textContent =
        "Input value: " +
        (item.input_value === null || item.input_value === undefined
            ? "Missing / imputed by model pipeline"
            : item.input_value);

    const track = document.createElement("div");
    track.className = "individual-shap-track";

    const bar = document.createElement("div");
    bar.className = "individual-shap-bar";

    const magnitude = Math.abs(validValue);
    const maxMagnitude = 0.15;
    const width = Math.min(100, magnitude / maxMagnitude * 100);

    bar.style.width = width + "%";
    bar.style.background =
        validValue > 0 ? "#dc2626" :
        validValue < 0 ? "#2563eb" : "#94a3b8";

    track.appendChild(bar);
    row.append(header, input, track);

    return row;
}

/* Render the current assessment's SHAP explanation */
function loadIndividualExplanation(result) {
    const container = document.getElementById("individualShapContent");

    // Older dashboards may not yet have this section.
    if (!container) {
        console.info(
            "Add the individualShapContent section to dashboard.html."
        );
        return;
    }

    container.replaceChildren();

    const explanation = result?.individual_explanation;

    if (!explanation?.available ||
        !Array.isArray(explanation.features)) {
        const message = document.createElement("p");
        message.textContent =
            "Individual SHAP explanation is unavailable for this result. " +
            "Submit a new assessment after confirming the Flask API returns SHAP values.";
        container.appendChild(message);
        return;
    }

    const baseline = Number(explanation.baseline_output);
    const output = Number(explanation.explained_output);

    const summary = document.createElement("div");
    summary.className = "individual-shap-summary";

    const baselineText = document.createElement("p");
    baselineText.textContent =
        "Baseline model output: " +
        (Number.isFinite(baseline)
            ? (baseline * 100).toFixed(2) + "%"
            : "Unavailable");

    const outputText = document.createElement("p");
    outputText.textContent =
        "This assessment's model output: " +
        (Number.isFinite(output)
            ? (output * 100).toFixed(2) + "%"
            : "Unavailable");

    summary.append(baselineText, outputText);
    container.appendChild(summary);

    const features = [...explanation.features].sort(
        (a, b) => Math.abs(Number(b.shap_value)) -
                  Math.abs(Number(a.shap_value))
    );

    const higher = features.filter(
        item => Number(item.shap_value) > 1e-8
    );

    const lower = features.filter(
        item => Number(item.shap_value) < -1e-8
    );

    const neutral = features.filter(
        item => Math.abs(Number(item.shap_value)) <= 1e-8
    );

    function addGroup(title, items, description) {
        const section = document.createElement("section");
        section.className = "individual-shap-group";

        const heading = document.createElement("h4");
        heading.textContent = title;

        const help = document.createElement("p");
        help.className = "individual-shap-description";
        help.textContent = description;

        section.append(heading, help);

        if (items.length === 0) {
            const empty = document.createElement("p");
            empty.textContent = "No features in this group.";
            section.appendChild(empty);
        } else {
            items.forEach(item => section.appendChild(createShapRow(item)));
        }

        container.appendChild(section);
    }

    addGroup(
        "Pushing the model output higher",
        higher,
        "Positive SHAP values move the model output upward relative to baseline."
    );

    addGroup(
        "Pushing the model output lower",
        lower,
        "Negative SHAP values move the model output downward relative to baseline."
    );

    if (neutral.length > 0) {
        addGroup(
            "Little or no contribution",
            neutral,
            "These features have approximately zero SHAP contribution for this assessment."
        );
    }

    const note = document.createElement("p");
    note.className = "individual-shap-notice";
    note.textContent =
        explanation.notice ||
        "SHAP explains the model output, not medical causation. " +
        "This is not a clinically validated personal heart-attack risk estimate.";

    container.appendChild(note);
}

/* Load dashboard data */
function loadDashboard() {
    const result = readStoredJSON(RESULT_KEY);

    const assessment =
        readStoredJSON(ASSESSMENT_KEY) ||
        readStoredJSON("cardioShieldData");

    if (!result) {
        setText("riskPercentage", "-- / 100");
        setText("riskCategory", "No assessment result");
        setText("heartRiskScore", "-- / 100");
        setText("heartRiskBadge", "Awaiting result");
        setText(
            "heartRiskMessage",
            "Complete your assessment to view the experimental model score."
        );
        setText(
            "simulationMessage",
            "The model is not re-run when these sliders change."
        );
        setText("simulationRisk", "Not calculated");

        loadIndividualExplanation(null);
        return;
    }

    const score = Number(
        result.model_score ??
        result.risk_percentage ?? result.probability_percentage
    );

    if (!Number.isFinite(score) || score < 0 || score > 100) {
        updateHeartRiskDisplay(NaN);
        loadIndividualExplanation(result);
        return;
    }

    updateHeartRiskDisplay(score);
    loadHealthSummary(assessment);
    updateSimulation();
    loadIndividualExplanation(result);
}

/* Initialize controls */
document.addEventListener("DOMContentLoaded", function () {
    ["activitySlider", "smokingSlider", "dietSlider"].forEach(function (id) {
        const slider = document.getElementById(id);
        if (slider) slider.addEventListener("input", updateSimulation);
    });

    const resetButton = document.getElementById("resetButton");

    if (resetButton) {
        resetButton.addEventListener("click", function () {
            const confirmed = window.confirm(
                "Clear the saved assessment result from this browser?"
            );

            if (!confirmed) return;

            localStorage.removeItem(RESULT_KEY);
            localStorage.removeItem(ASSESSMENT_KEY);
            localStorage.removeItem("cardioShieldData");
            localStorage.removeItem("cardioShieldAssessmentCompleted");

            window.location.reload();
        });
    }

    loadDashboard();
});
