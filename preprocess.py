
"""
CardioShield-AgeLess
Dataset preprocessing and validation

Input:
    datasets/cardioshield_combined_heart_disease.csv

Output:
    datasets/cardioshield_preprocessed.csv

Note:
    Missing values are intentionally preserved here.
    Imputation should be fitted on training data only.
"""

from pathlib import Path

import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets"

INPUT_FILE = DATASET_DIR / "cardioshield_combined_heart_disease.csv"
OUTPUT_FILE = DATASET_DIR / "cardioshield_preprocessed.csv"

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


# ==================================================
# 2. LOAD DATASET
# ==================================================

def load_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}\n\n"
            "Check that the CSV file exists inside the datasets folder."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    print("\n========== CARDIOSHIELD PREPROCESSING ==========")
    print(f"Input file: {INPUT_FILE}")
    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    return df


# ==================================================
# 3. VALIDATE REQUIRED COLUMNS
# ==================================================

def validate_columns(df):
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the dataset:\n"
            + ", ".join(missing_columns)
            + "\n\nCheck the dataset column names before continuing."
        )

    print("\nColumn validation: PASSED")
    print(f"Model input features: {len(FEATURES)}")
    print(f"Target column: {TARGET}")


# ==================================================
# 4. CLEAN COLUMN VALUES
# ==================================================

def clean_values(df):
    df = df.copy()

    # Keep only the features and target required by the model.
    # source_dataset and unrelated columns are excluded.
    df = df[FEATURES + [TARGET]].copy()

    # Convert model columns to numeric values.
    # Unexpected text becomes missing and is reported below.
    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where the target is unavailable.
    before = len(df)

    df = df.dropna(subset=[TARGET]).copy()

    removed_target_rows = before - len(df)

    if removed_target_rows:
        print(
            "Rows removed because target was missing:",
            removed_target_rows
        )

    # The target must contain only binary labels 0 and 1.
    valid_target_values = {0, 1}

    actual_target_values = set(df[TARGET].unique())

    invalid_target_values = (
        actual_target_values - valid_target_values
    )

    if invalid_target_values:
        raise ValueError(
            f"Unexpected target labels: {invalid_target_values}. "
            "Expected only 0 and 1."
        )

    # Remove exact duplicate records.
    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    removed_duplicates = before - len(df)

    print("Duplicate rows removed:", removed_duplicates)

    # Treat physically implausible zero measurements as missing.
    # These fields represent positive clinical measurements.
    # The original input CSV is not modified.
    for column in ["resting_bp", "cholesterol"]:
        zero_count = int((df[column] == 0).sum())

        if zero_count:
            df.loc[df[column] == 0, column] = float("nan")

            print(
                f"{column}: converted {zero_count} zero value(s) "
                "to missing values for later imputation."
            )

    return df


# ==================================================
# 5. REPORT MISSING VALUES
# ==================================================

def report_missing_values(df):
    print("\n========== MISSING VALUES ==========")

    missing_counts = df[FEATURES].isna().sum()

    report = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percentage": (
            missing_counts / len(df) * 100
        ).round(2),
    })

    print(report.to_string())

    print(
        "\nMissing values are preserved. "
        "The training pipeline must impute them using training data only."
    )


# ==================================================
# 6. REPORT TARGET DISTRIBUTION
# ==================================================

def report_target(df):
    print("\n========== TARGET DISTRIBUTION ==========")

    counts = df[TARGET].value_counts().sort_index()

    for label, count in counts.items():
        percentage = count / len(df) * 100

        print(
            f"Class {int(label)}: {count} rows "
            f"({percentage:.2f}%)"
        )

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "The target must contain both classes 0 and 1."
        )


# ==================================================
# 7. SAVE MODEL-READY DATASET
# ==================================================

def save_dataset(df):
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== OUTPUT ==========")
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Saved file: {OUTPUT_FILE}")
    print("\nPreprocessing completed successfully.")


# ==================================================
# 8. MAIN
# ==================================================

def main():
    df = load_dataset()

    validate_columns(df)

    df = clean_values(df)

    report_missing_values(df)

    report_target(df)

    save_dataset(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nPREPROCESSING FAILED")
        print(error)
        raise
