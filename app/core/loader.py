import pandas as pd
import os
from app.config import DATA_PATH

# Canonical column names matching your Excel schema
EXPECTED_COLUMNS = [
    "Posting Date",
    "Site ID",
    "Invoice Account",
    "Name",
    "State Name",
    "Customer account",
    "Name2",
    "Item ID",
    "Item Name",
    "Base Quantity",
    "MRP",
    "Unit Price",
    "Pack Type",
    "Pack Type Group",
    "Pack Size",
    "Product Segment",
    "Brand",
    "Segment (Customer)",
    "Sub segment (Customer)",
    "Customer group",
    "GM",
    "SM",
    "ASM",
    "FE",
    "Salesman",
    "PSR",
]

# Columns to use for full-text search (the most query-relevant ones)
SEARCHABLE_COLUMNS = [
    "Name",
    "Name2",
    "State Name",
    "Site ID",
    "Item Name",
    "Brand",
    "Product Segment",
    "Pack Type",
    "Pack Type Group",
    "Segment (Customer)",
    "Sub segment (Customer)",
    "Customer group",
    "GM",
    "SM",
    "ASM",
    "FE",
    "Salesman",
    "PSR",
]

# Numeric columns to convert to float at load time
# so the LLM never has to deal with string-typed numbers
NUMERIC_COLS = [
    "Line Amount",
    "Total Amount",
    "MRP",
    "Unit Price",
    "Base Quantity",
    "IGST amount",
    "CGST AMOUNT",
    "SGST AMOUNT",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load and clean the vendor Excel file.
    Returns a normalized DataFrame ready for search.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif ext == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Normalize column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Warn about missing expected columns
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[loader] WARNING: Missing expected columns: {missing}")

    # Fill NaN with empty string to avoid LLM prompt pollution
    df = df.fillna("")

    # Strip whitespace from all string cells
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Normalize key text columns to Title Case for consistent matching
    title_case_cols = [
        "Name", "Name2", "State Name", "Item Name", "Brand",
        "Product Segment", "Pack Type", "Pack Type Group",
        "Segment (Customer)", "Sub segment (Customer)",
        "Customer group", "GM", "SM", "ASM", "FE", "Salesman", "PSR"
    ]
    for col in title_case_cols:
        if col in df.columns:
            df[col] = df[col].str.title()

    # Normalize date column
    if "Posting Date" in df.columns:
        df["Posting Date"] = pd.to_datetime(
            df["Posting Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d").fillna("")

    # ── Convert numeric columns to float ─────────────────────────────────────
    # Do this AFTER stripping whitespace and commas.
    # Columns are already actual floats — the LLM never needs pd.to_numeric().
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", "").str.strip(),
                errors="coerce"
            )

    # Build a searchable text blob per row for fuzzy matching
    search_cols = [c for c in SEARCHABLE_COLUMNS if c in df.columns]
    df["_search_text"] = df[search_cols].apply(
        lambda row: " ".join(v for v in row if v), axis=1
    ).str.lower()

    print(f"[loader] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df