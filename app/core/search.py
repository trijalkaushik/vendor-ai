import re
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process
import ollama

# ── Model config ──────────────────────────────────────────────────────────────
MODEL = "llama3.2:3b"

# ── Column keywords for direct matching (no LLM needed) ──────────────────────
# If the query contains these words, filter on that column
COLUMN_KEYWORDS = {
    "State Name":          ["state", "states"],
    "Site ID":             ["site", "depot", "warehouse"],
    "Item Name":           ["item", "product", "items", "products"],
    "Brand":               ["brand", "brands"],
    "Product Segment":     ["segment", "category"],
    "Pack Type":           ["pack", "packs", "packaging"],
    "Salesman":            ["salesman", "sales man", "salesmen"],
    "ASM":                 ["asm"],
    "SM":                  ["sm"],
    "GM":                  ["gm"],
    "PSR":                 ["psr"],
    "FE":                  ["fe"],
    "Customer group":      ["group"],
    "Posting Date":        ["date", "posted", "posting"],
}

# Fields the user might want to retrieve
RETRIEVAL_KEYWORDS = {
    "Customer account":         ["customer account", "account number", "account no"],
    "Invoice Account":          ["invoice account"],
    "Name":                     ["name", "vendor", "customer"],
    "MRP":                      ["mrp", "price", "cost"],
    "Unit Price":               ["unit price", "selling price"],
    "Base Quantity":            ["quantity", "qty", "units"],
    "Salesman":                 ["salesman"],
    "ASM":                      ["asm"],
    "PSR":                      ["psr"],
    "State Name":               ["state"],
    "Site ID":                  ["site", "depot"],
}

# ── Answer prompt (only called once, for formatting) ─────────────────────────
ANSWER_PROMPT = """You are a sales data assistant. Be concise and direct.

User asked: "{query}"

Matching records:
{records}

Answer the question using only the data above. If multiple records, list them.
Do not add any information not present in the records."""


def _call_llm(prompt: str, max_tokens: int = 250) -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "temperature": 0.0},
    )
    return resp["message"]["content"].strip()


# ── Pure Python intent extraction (no LLM) ───────────────────────────────────

def extract_intent(query: str) -> dict:
    """
    Extract search intent using simple string matching — no LLM.
    Fast, deterministic, and doesn't misread 'a b traders' as salesman 'B'.
    """
    q = query.lower().strip()

    # Strip common filler words to isolate the entity name
    # e.g. "pull the customer account for a b traders" -> "a b traders"
    filler_patterns = [
        r"^(pull|get|show|find|fetch|give|list|what is|what are|tell me)\s+(the\s+|me\s+|all\s+)?",
        r"\b(for|of|from|in|at|by|with|about|related to|belonging to)\b\s+",
        r"\b(customer account|invoice account|account number|account no|account)\b\s*",
        r"\b(vendor|salesman|asm|psr|sm|gm|fe|site|depot|state|brand|item|product|price|mrp|quantity|qty)\b\s*",
        r"\b(details|info|information|data|records?)\b\s*",
        r"\b(please|kindly|can you|could you)\b\s*",
    ]

    entity = q
    for pat in filler_patterns:
        entity = re.sub(pat, "", entity, flags=re.IGNORECASE).strip()
    entity = entity.strip(" ,.-")

    # Detect which field the user wants to retrieve
    want_fields = []
    for field, keywords in RETRIEVAL_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            want_fields.append(field)

    # Detect any column-specific filters (e.g. "in Delhi", "salesman Ramesh")
    col_filters = {}
    for col, keywords in COLUMN_KEYWORDS.items():
        for kw in keywords:
            pattern = rf"\b{re.escape(kw)}\s+([A-Za-z0-9 ]+)"
            match = re.search(pattern, q)
            if match:
                col_filters[col] = match.group(1).strip().title()
                break

    result = {
        "entity": entity.title() if entity else None,
        "want_fields": want_fields,
        "col_filters": col_filters,
    }
    print(f"[search] Intent: {result}")
    return result


def pandas_filter(df: pd.DataFrame, intent: dict) -> pd.DataFrame:
    """Apply filters from intent — entity matches Name/Name2, plus any col_filters."""
    result = df.copy()

    # Match entity name against Name and Name2
    entity = intent.get("entity")
    if entity:
        entity_lower = entity.lower()
        name_cols = [c for c in ["Name", "Name2"] if c in result.columns]
        if name_cols:
            mask = result[name_cols].apply(
                lambda col: col.str.lower().str.contains(entity_lower, na=False)
            ).any(axis=1)
            result = result[mask]

    # Apply any column-specific filters
    for col, val in intent.get("col_filters", {}).items():
        if col in result.columns:
            result = result[
                result[col].str.lower().str.contains(val.lower(), na=False)
            ]

    return result


def fuzzy_search(df: pd.DataFrame, query: str, top_n: int = 10, threshold: int = 50) -> pd.DataFrame:
    """Fuzzy fallback against the _search_text blob."""
    if "_search_text" not in df.columns or df.empty:
        return df.head(top_n)

    scores = process.extract(
        query.lower(),
        df["_search_text"].tolist(),
        scorer=fuzz.WRatio,
        limit=top_n,
    )
    indices = [s[2] for s in scores if s[1] >= threshold]
    if not indices:
        return pd.DataFrame()
    return df.iloc[indices]


def _get_candidates(df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, dict]:
    """Returns (candidates_df, intent_dict)."""
    intent = extract_intent(query)
    candidates = pandas_filter(df, intent)
    print(f"[search] After pandas filter: {len(candidates)} rows")

    if candidates.empty:
        print("[search] Falling back to fuzzy search...")
        entity = intent.get("entity") or query
        candidates = fuzzy_search(df, entity)
        print(f"[search] After fuzzy search: {len(candidates)} rows")

    if candidates.empty:
        candidates = fuzzy_search(df, query)

    return candidates, intent


def _format_records(candidates: pd.DataFrame, intent: dict) -> str:
    """Only include columns the user actually asked about — keeps LLM prompt tiny."""
    want = intent.get("want_fields", [])

    # Always include Name for context
    base_cols = ["Name", "State Name"]
    target_cols = list(dict.fromkeys(base_cols + want))  # preserve order, no dupes
    
    # If no specific fields requested, send a small default set
    if not want:
        target_cols = ["Name", "Name2", "State Name", "Site ID",
                       "Customer account", "Invoice Account",
                       "Item Name", "Brand", "Salesman", "ASM", "PSR"]

    cols_to_use = [c for c in target_cols if c in candidates.columns]
    slim = candidates[cols_to_use].head(5)
    return slim.to_json(orient="records", indent=None)


def generate_answer(query: str, candidates: pd.DataFrame, intent: dict) -> str:
    """Single LLM call — just for formatting the answer."""
    if candidates.empty:
        return "No matching vendor records found."

    records_str = _format_records(candidates, intent)
    print(f"[search] Sending {min(len(candidates), 5)} rows to LLM...")
    prompt = ANSWER_PROMPT.format(query=query, records=records_str)
    return _call_llm(prompt, max_tokens=250)


# ── Public API ────────────────────────────────────────────────────────────────

def search(df: pd.DataFrame, query: str) -> Optional[str]:
    """Full pipeline — returns a natural language answer string."""
    if not query.strip():
        return None
    candidates, intent = _get_candidates(df, query)
    if candidates.empty:
        return "No matching vendor records found."
    return generate_answer(query, candidates, intent)


def search_raw(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Returns a DataFrame of top matches without LLM answer generation."""
    candidates, _ = _get_candidates(df, query)
    display_cols = [c for c in candidates.columns if not c.startswith("_")]
    return candidates[display_cols].head(10)