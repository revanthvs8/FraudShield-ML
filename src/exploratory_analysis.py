import pandas as pd
from pathlib import Path

# Locate the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the dataset path
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Check whether the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place creditcard.csv inside the data folder."
    )

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Display the first five rows
print("\n--- FIRST FIVE TRANSACTIONS ---")
print(df.head())

# Display the number of rows and columns
print("\n--- DATASET SHAPE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Check for missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Count legitimate and fraudulent transactions
print("\n--- TRANSACTION CLASS COUNTS ---")
print(df["Class"].value_counts())

# Display the percentage of each class
print("\n--- TRANSACTION CLASS PERCENTAGES ---")
print(
    (df["Class"].value_counts(normalize=True) * 100).round(4)
)

# Display dataset information
print("\n--- DATASET INFORMATION ---")
df.info()