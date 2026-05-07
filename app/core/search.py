import re
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

# ── LLM clients ───────────────────────────────────────────────────────────────
from app.config import MODEL, GROQ_API_KEY, GROQ_MODEL

_groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"[llm] Using Groq ({GROQ_MODEL})")
    except ImportError:
        print("[llm] groq package not installed — falling back to Ollama")
else:
    print(f"[llm] No GROQ_API_KEY found — using Ollama ({MODEL})")


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL — Groq primary, Ollama fallback
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 250) -> str:
    if _groq_client:
        try:
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[llm] Groq error: {e} — falling back to Ollama")

    # Ollama fallback
    import ollama
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "temperature": 0.0, "num_ctx": 2048},
    )
    return resp["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_KEYWORDS = {
    "State Name":      ["state", "states"],
    "Site ID":         ["site", "depot", "warehouse"],
    "Item Name":       ["item", "product", "items", "products"],
    "Brand":           ["brand", "brands"],
    "Product Segment": ["segment", "category"],
    "Pack Type":       ["pack", "packs", "packaging"],
    "Salesman":        ["salesman", "sales man", "salesmen"],
    "ASM":             ["asm"],
    "SM":              ["sm"],
    "GM":              ["gm"],
    "PSR":             ["psr"],
    "FE":              ["fe"],
    "Customer group":  ["group"],
    "Posting Date":    ["date", "posted", "posting"],
    "Source City":     ["city"],
    "Source State":    ["source state"],
}

RETRIEVAL_KEYWORDS = {
    "Customer account": ["customer account", "account number", "account no"],
    "Invoice Account":  ["invoice account"],
    "Name":             ["name", "vendor", "customer"],
    "MRP":              ["mrp", "price", "cost"],
    "Unit Price":       ["unit price", "selling price"],
    "Base Quantity":    ["quantity", "qty", "units"],
    "Salesman":         ["salesman"],
    "ASM":              ["asm"],
    "PSR":              ["psr"],
    "State Name":       ["state"],
    "Site ID":          ["site", "depot"],
    "Customer GST No.": ["gst", "gstin", "gst number", "gst no"],
    "Source City":      ["city"],
    "Source State":     ["source state"],
    "Line Amount":      ["line amount", "amount"],
    "Total Amount":     ["total", "total amount"],
}

LIST_KEYWORDS = [
    "all", "everything", "every", "list", "show all", "get all",
    "full", "complete", "details", "transactions", "records", "history",
]

LOOKUP_KEYWORDS = [
    "gst", "gstin", "account", "customer account", "invoice account",
    "salesman", "asm", "psr", "sm", "gm", "fe", "site", "depot",
    "state", "city", "mrp", "price", "unit price",
]

ANALYTICAL_KEYWORDS = [
    "top", "bottom", "most", "least", "total", "sum", "average", "avg",
    "best", "worst", "highest", "lowest", "rank", "ranking", "compare",
    "how many", "count", "which vendor", "which brand", "which item",
    "belongs to", "who has", "who is", "maximum", "minimum", "max", "min",
    "per state", "per brand", "per city", "breakdown", "distribution",
    "revenue", "performance", "underperform", "overperform",
]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PANDAS_PROMPT = """You are a Python/pandas expert. You have a DataFrame called `df`.

Column names and sample values:
{schema}

Important notes:
- Numeric columns like Line Amount, MRP, Unit Price, Base Quantity are already float. Use them directly.
- NEVER use pd.to_numeric or .astype() — columns are already the right type.
- NEVER call .nlargest() or .nsmallest() on string columns.
- String comparisons must be case-insensitive: use .str.lower().str.contains(...)
- Always include vendor Name column in results where possible.
- For GST reverse lookup, filter on 'Customer GST No.' column.

The user asked: "{query}"

Write ONE Python expression that answers this. Return ONLY the expression.
No explanation. No imports. No markdown. No variable assignments.
The expression must evaluate to a DataFrame, Series, or scalar.

Examples:
"top 10 sales by amount"          → df.nlargest(10,'Line Amount')[['Name','Item Name','Line Amount','State Name']]
"total sales in Jharkhand"        → df[df['State Name'].str.lower().str.contains('jharkhand',na=False)]['Line Amount'].sum()
"total sales by vendor Jharkhand" → df[df['State Name'].str.lower().str.contains('jharkhand',na=False)].groupby('Name')['Line Amount'].sum().reset_index(name='Total Sales').sort_values('Total Sales',ascending=False)
"which vendor has GST 20XYZ"      → df[df['Customer GST No.'].str.lower().str.contains('20xyz',na=False)][['Name','State Name','Customer account','Customer GST No.']].drop_duplicates('Name')
"top 5 brands by quantity"        → df.groupby('Brand')['Base Quantity'].sum().reset_index(name='Total Qty').nlargest(5,'Total Qty')
"count vendors per state"         → df.groupby('State Name')['Name'].nunique().reset_index(name='Vendor Count').sort_values('Vendor Count',ascending=False)
"compare A B Traders vs A S Grow" → df[df['Name'].str.lower().str.contains('a b traders|a s grow',na=False)].groupby('Name')[['Line Amount','Base Quantity']].sum().reset_index()

Expression:"""

FORMAT_PROMPT = """You are a sales data assistant. Be clear and concise.

The user asked: "{query}"

Query result:
{result}

Explain this result in plain English in 1-2 sentences. If it's a number, say what it means. Be brief."""

ANSWER_PROMPT = """You are a sales data assistant. Be strictly factual.

User asked: "{query}"

Matching records:
{records}

RULES:
- Only describe what is in the records above.
- If the Name field does not match what the user asked for, say so explicitly.
- Never invent or rename vendors.
- Be concise."""


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_hint(df: pd.DataFrame) -> str:
    priority_cols = [
        "Name", "State Name", "Site ID", "Customer account", "Invoice Account",
        "Customer GST No.", "Item Name", "Item ID", "Brand", "Product Segment",
        "Pack Type", "Pack Type Group", "Pack Size", "Base Quantity",
        "MRP", "Unit Price", "Line Amount", "Total Amount",
        "Salesman", "ASM", "SM", "GM", "FE", "PSR",
        "Source City", "Source State", "Posting Date",
    ]
    ordered = [c for c in priority_cols if c in df.columns]
    rest = [c for c in df.columns if c not in ordered and not c.startswith("_")]
    lines = []
    for col in ordered + rest:
        samples = df[col].dropna().replace("", pd.NA).dropna().head(3).tolist()
        dtype = str(df[col].dtype)
        lines.append(f"  '{col}' ({dtype}): e.g. {samples}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# QUERY ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def _is_analytical(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in ANALYTICAL_KEYWORDS)


def _is_lookup(query: str) -> bool:
    q = query.lower()
    if any(kw in q for kw in LIST_KEYWORDS):
        return False
    return any(kw in q for kw in LOOKUP_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT-TO-PANDAS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _clean_code(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```[a-z]*\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def _result_to_str(result) -> str:
    if isinstance(result, pd.DataFrame):
        return "Empty result." if result.empty else result.head(20).to_string(index=False)
    elif isinstance(result, pd.Series):
        return result.head(20).to_string()
    elif isinstance(result, float):
        return f"{result:,.2f}"
    else:
        return str(result)


def _result_to_rows(result) -> list:
    if isinstance(result, pd.DataFrame):
        return result.head(100).fillna("").to_dict(orient="records")
    elif isinstance(result, pd.Series):
        return result.head(100).reset_index().fillna("").to_dict(orient="records")
    else:
        return [{"Result": result}]


# Cache schema so it's not rebuilt every query
_schema_cache: str = ""

def _get_schema(df: pd.DataFrame) -> str:
    global _schema_cache
    if not _schema_cache:
        _schema_cache = build_schema_hint(df)
    return _schema_cache


def run_analytical_query(df: pd.DataFrame, query: str) -> dict:
    schema = _get_schema(df)
    code_prompt = PANDAS_PROMPT.format(schema=schema, query=query)

    print(f"[analytical] Generating code for: {query}")
    code = _clean_code(_call_llm(code_prompt, max_tokens=150))
    print(f"[analytical] Code: {code}")

    try:
        result = eval(code, {"df": df.copy(), "pd": pd})
    except Exception as e:
        print(f"[analytical] Error: {e} — retrying...")
        retry_prompt = (
            f"{code_prompt}\n\nPrevious attempt:\n{code}\n"
            f"Error: {e}\nFix it. Return only the corrected expression:"
        )
        code = _clean_code(_call_llm(retry_prompt, max_tokens=150))
        print(f"[analytical] Retry code: {code}")
        try:
            result = eval(code, {"df": df.copy(), "pd": pd})
        except Exception as e2:
            return {"answer": f"Could not compute that query. Error: {e2}", "rows": [], "code": code}

    rows = _result_to_rows(result)
    result_str = _result_to_str(result)

    # Skip second LLM call for simple results — return directly
    if isinstance(result, (int, float)):
        answer = f"**{result_str}**"
    elif isinstance(result, pd.DataFrame) and not result.empty:
        # Only call LLM for formatting if Groq is available (fast)
        # otherwise just return a plain summary to avoid double wait
        if _groq_client:
            format_prompt = FORMAT_PROMPT.format(query=query, result=result_str)
            answer = _call_llm(format_prompt, max_tokens=150)
        else:
            answer = f"Found {len(result)} records. See the table →"
    else:
        answer = result_str

    return {"answer": answer, "rows": rows, "code": code}


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_intent(query: str) -> dict:
    q = query.lower().strip()

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

    want_fields = []
    for field, keywords in RETRIEVAL_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            want_fields.append(field)

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
        "is_lookup": _is_lookup(q),
    }
    print(f"[search] Intent: {result}")
    return result


def pandas_filter(df: pd.DataFrame, intent: dict) -> pd.DataFrame:
    result = df.copy()
    entity = intent.get("entity")
    if entity:
        entity_lower = entity.lower()
        name_cols = [c for c in ["Name", "Name2"] if c in result.columns]
        if name_cols:
            mask = result[name_cols].apply(
                lambda col: col.str.lower().str.contains(
                    re.escape(entity_lower), na=False
                )
            ).any(axis=1)
            result = result[mask]

    for col, val in intent.get("col_filters", {}).items():
        if col in result.columns:
            result = result[
                result[col].str.lower().str.contains(val.lower(), na=False)
            ]
    return result


def fuzzy_search(df: pd.DataFrame, query: str, top_n: int = 10, threshold: int = 72) -> pd.DataFrame:
    if df.empty:
        return df

    name_col  = "Name"  if "Name"  in df.columns else None
    name2_col = "Name2" if "Name2" in df.columns else None

    if name_col:
        names = df[name_col].fillna("").tolist()
        scores = process.extract(query.lower(), [n.lower() for n in names],
                                 scorer=fuzz.WRatio, limit=top_n * 3)
        score_map = {}
        for _m, score, idx in scores:
            if score >= threshold:
                score_map[idx] = max(score_map.get(idx, 0), score)

        if name2_col:
            names2 = df[name2_col].fillna("").tolist()
            scores2 = process.extract(query.lower(), [n.lower() for n in names2],
                                      scorer=fuzz.WRatio, limit=top_n * 3)
            for _m, score, idx in scores2:
                if score >= threshold:
                    score_map[idx] = max(score_map.get(idx, 0), score)

        if score_map:
            sorted_indices = sorted(score_map, key=score_map.get, reverse=True)
            seen_names, deduped = set(), []
            for idx in sorted_indices:
                name = df.iloc[idx][name_col].lower()
                if name not in seen_names:
                    seen_names.add(name)
                    deduped.append(idx)
                if len(deduped) >= top_n:
                    break
            result = df.iloc[deduped]
            print(f"[search] Fuzzy matched: {[df.iloc[i][name_col] for i in deduped[:3]]}")
            return result

    if "_search_text" in df.columns:
        blob = process.extract(query.lower(), df["_search_text"].tolist(),
                               scorer=fuzz.WRatio, limit=top_n)
        indices = [s[2] for s in blob if s[1] >= threshold]
        if indices:
            return df.iloc[indices]

    return pd.DataFrame()


def token_search(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    if not entity or "Name" not in df.columns:
        return pd.DataFrame()
    tokens = [t.lower() for t in entity.lower().split() if len(t) > 1]
    if not tokens:
        return pd.DataFrame()
    mask = df["Name"].str.lower().apply(lambda name: all(tok in name for tok in tokens))
    result = df[mask]
    if not result.empty:
        print(f"[search] Token search: {len(result)} rows for {tokens}")
    return result


def _get_candidates(df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, dict]:
    intent = extract_intent(query)
    entity = intent.get("entity") or query

    candidates = pandas_filter(df, intent)
    print(f"[search] After pandas filter: {len(candidates)} rows")

    if candidates.empty and entity:
        candidates = token_search(df, entity)

    if candidates.empty:
        print("[search] Fuzzy search (72)...")
        candidates = fuzzy_search(df, entity, threshold=72)
        print(f"[search] Fuzzy (72): {len(candidates)} rows")

    if candidates.empty:
        print("[search] Fuzzy search (60)...")
        candidates = fuzzy_search(df, entity, threshold=60)
        print(f"[search] Fuzzy (60): {len(candidates)} rows")

    return candidates, intent


def _mismatch_prefix(query_entity: str, candidates: pd.DataFrame) -> str:
    if not query_entity or "Name" not in candidates.columns or candidates.empty:
        return ""
    top_name = candidates.iloc[0]["Name"]
    score = fuzz.WRatio(query_entity.lower(), top_name.lower())
    if score < 70:
        return f"⚠️ No exact match for '{query_entity}'. Closest result is '{top_name}'.\n\n"
    return ""


def _format_records(candidates: pd.DataFrame, intent: dict) -> str:
    want = intent.get("want_fields", [])
    base_cols = ["Name", "State Name"]
    target_cols = list(dict.fromkeys(base_cols + want))

    if not want:
        target_cols = [
            "Name", "Name2", "State Name", "Site ID",
            "Customer account", "Invoice Account",
            "Customer GST No.",
            "Item Name", "Brand", "Salesman", "ASM", "PSR",
        ]

    cols_to_use = [c for c in target_cols if c in candidates.columns]
    slim = candidates[cols_to_use]

    if intent.get("is_lookup") and "Name" in slim.columns:
        slim = slim.drop_duplicates(subset=["Name"])
        print(f"[search] Lookup — {len(slim)} unique vendors")

    return slim.head(100).to_json(orient="records", indent=None)


def generate_answer(query: str, candidates: pd.DataFrame, intent: dict) -> str:
    if candidates.empty:
        return "No matching vendor records found."
    prefix = _mismatch_prefix(intent.get("entity", ""), candidates)
    records_str = _format_records(candidates, intent)
    prompt = ANSWER_PROMPT.format(query=query, records=records_str)
    return prefix + _call_llm(prompt, max_tokens=300)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def search(df: pd.DataFrame, query: str) -> Optional[str]:
    if not query.strip():
        return None
    if _is_analytical(query):
        print("[router] Analytical → text-to-pandas")
        return run_analytical_query(df, query)["answer"]
    candidates, intent = _get_candidates(df, query)
    if candidates.empty:
        return f"No matching records found for '{intent.get('entity') or query}'."
    return generate_answer(query, candidates, intent)


def search_raw(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()
    if _is_analytical(query):
        print("[router] Analytical → text-to-pandas (table)")
        rows = run_analytical_query(df, query).get("rows", [])
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    candidates, intent = _get_candidates(df, query)
    if candidates.empty:
        return pd.DataFrame()
    display_cols = [c for c in candidates.columns if not c.startswith("_")]
    result = candidates[display_cols]
    if intent.get("is_lookup") and "Name" in result.columns:
        result = result.drop_duplicates(subset=["Name"])
    return result.head(100)