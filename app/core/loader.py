# loads cleaned.xlsx
# creates search_text column

import pandas as pd

DATA_PATH = "data/processed/cleaned.xlsx"


def load_data():
    df = pd.read_excel(DATA_PATH).fillna("")

    # Build searchable text
    df["search_text"] = df.apply(lambda row: build_text(row), axis=1)

    return df


def build_text(row):
    return f"""
    Vendor: {row.get('Name','')}
    City: {row.get('Source City','')}
    State: {row.get('Source State','')}
    GST: {row.get('Customer GST No.','')}
    Account: {row.get('Customer account','')}
    
    Product: {row.get('Item Name','')}
    Brand: {row.get('Brand','')}
    Segment: {row.get('Product Segment','')}
    
    Category: Beverage distribution
    """