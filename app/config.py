# app/config.py
import os

MODEL = os.environ.get("VENDOR_MODEL", "llama3.2:3b")
DATA_PATH = os.environ.get("VENDOR_DATA_PATH", "data/processed/cleaned.xlsx")
FUZZY_THRESHOLD = int(os.environ.get("FUZZY_THRESHOLD", "45"))