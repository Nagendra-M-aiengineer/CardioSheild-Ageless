import pandas as pd

# Load the combined CardioShield dataset
file_path = "datasets/cardioshield_combined_heart_disease.csv"

df = pd.read_csv(file_path)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(column)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["heart_disease"].value_counts(dropna=False))

print("\n========== BASIC STATISTICS ==========")
print(df.describe())