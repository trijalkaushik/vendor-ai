import pandas as pd
import os

RAW_PATH = "C:/Users/admin/vendor-ai/data/raw/salesdata2.xlsx"
OUTPUT_PATH = "C:/Users/admin/vendor-ai/data/processed/cleaned.xlsx"


def clean_excel():
    print("Loading Excel...")
    df = pd.read_excel(RAW_PATH)

    print("Initial rows:", len(df))

    # ---------------- CLEANING ----------------

    # Remove completely empty rows
    df = df.dropna(how="all")
    # Strip spaces from all values
    df = df.apply(lambda col: col.map(lambda x: str(x).strip() if pd.notna(x) else x))

    # Remove rows with no Name (important)
    df = df[df["Name"].notna()]
    df = df[df["Name"] != ""]

    # Optional: remove duplicates
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    # ---------------- SAVE ----------------
    os.makedirs("data/processed", exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)

    print("Cleaned rows:", len(df))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_excel()