import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

MODEL        = os.environ.get("VENDOR_MODEL",     "llama3.2:3b")
DATA_PATH    = os.environ.get("VENDOR_DATA_PATH", "data/processed/cleaned.xlsx")
RAW_PATH     = os.environ.get("VENDOR_RAW_PATH",  "data/raw/salesdata2.xlsx")
FUZZY_THRESHOLD = int(os.environ.get("FUZZY_THRESHOLD", "45"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY",     "")
GROQ_MODEL   = "llama-3.1-8b-instant"   # fast + smarter than local 3b