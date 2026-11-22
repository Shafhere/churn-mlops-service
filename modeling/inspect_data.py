
import hashlib
from pathlib import Path

import pandas as pd

# Location of our original dataset
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")


def main():
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(DATA_PATH)

    # 1. Understand the dataset
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    # 2. Check missing values
    missing = df.isna().sum().sum()
    print("\nMissing cells:", missing)

    # 3. Check values that cannot be converted to numbers
    charges = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    print("Invalid TotalCharges:", charges.isna().sum())

    # 4. Check duplicate customer identifiers
    duplicates = df["customerID"].duplicated().sum()
    print("Duplicate customer IDs:", duplicates)

    # 5. Check target class distribution
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())

    # 6. Generate a fingerprint of the original CSV
    file_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    print("\nDataset SHA-256:", file_hash)


if __name__ == "__main__":
    main()
