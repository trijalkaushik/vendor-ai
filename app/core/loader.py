# loads cleaned.xlsx
# creates search_text column

import pandas as pd

DATA_PATH = "data/processed/cleaned.xlsx"


def load_data():
    df = pd.read_excel(DATA_PATH).fillna("")

    # Build searchable text
    df["search_text"] = df.apply(lambda row: build_text(row), axis=1)

    return df


IMPORTANT_COLUMNS = [
    "Name",
    "Customer account",
    "State Name",
    "Item Name",
    "Brand",
    "MRP",
    "Unit Price"
]

def build_text(row):
    parts = []

    for col, val in row.items():
        if val and str(val).strip() != "":
            text = str(val)

            if col in IMPORTANT_COLUMNS:
                parts.append(text)
                parts.append(text)  # boost weight
            else:
                parts.append(text)

    return " ".join(parts)