import pandas as pd
import os

# Use paths relative to project root, not hardcoded absolute paths
RAW_PATH = os.environ.get("VENDOR_RAW_PATH", "data/raw/salesdata2.xlsx")
OUTPUT_PATH = os.environ.get("VENDOR_DATA_PATH", "data/processed/cleaned.xlsx")


def clean_excel(raw_path: str = RAW_PATH, output_path: str = OUTPUT_PATH):
    print(f"[cleaner] Loading: {raw_path}")
    df = pd.read_excel(raw_path, dtype=str)

    print(f"[cleaner] Initial rows: {len(df)}")

    # Strip column name whitespace
    df.columns = df.columns.str.strip()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Strip whitespace from all cell values
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    # Replace bare "nan" strings left over from dtype=str conversion
    df = df.replace("nan", "")

    # Remove rows with no Name
    df = df[df["Name"].notna()]
    df = df[df["Name"].str.strip() != ""]

    # Remove duplicates
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"[cleaner] Cleaned rows: {len(df)}")
    print(f"[cleaner] Saved to: {output_path}")
    return df


if __name__ == "__main__":
    clean_excel()