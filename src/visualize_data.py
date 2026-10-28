import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate project folder and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

# Create reports folder if it does not exist
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Set chart appearance
sns.set_theme(style="whitegrid")

# ---------------------------------------
# Chart 1: Legitimate vs Fraudulent
# Transactions
# ---------------------------------------

class_counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x=["Legitimate", "Fraudulent"],
    y=class_counts.values
)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")

plt.tight_layout()
plt.savefig(REPORTS_PATH / "transaction_classes.png", dpi=150)
plt.show()

# ---------------------------------------
# Chart 2: Transaction Amount Distribution
# ---------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50,
    element="step",
    stat="count",
    common_norm=False
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.savefig(REPORTS_PATH / "transaction_amounts.png", dpi=150)
plt.show()

# ---------------------------------------
# Chart 3: Fraud Percentage
# ---------------------------------------

fraud_percentage = df["Class"].value_counts(normalize=True) * 100

print("\n--- FRAUD ANALYSIS ---")
print(f"Total transactions: {len(df):,}")
print(f"Fraudulent transactions: {(df['Class'] == 1).sum():,}")
print(f"Legitimate transactions: {(df['Class'] == 0).sum():,}")
print(f"Fraud percentage: {fraud_percentage.get(1, 0):.4f}%")

print("\nCharts saved in:", REPORTS_PATH)