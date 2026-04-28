from rapidfuzz import fuzz

IMPORTANT_FIELDS = [
    "Name",
    "Customer account",
    "Customer GST No.",
    "Source City",
    "Source State",
    "Item Name",
    "Brand"
]


def normalize_query(query):
    return query.lower().strip()


# ---------------- INTENT DETECTION ----------------

def extract_filters(df, query):
    q = normalize_query(query)

    cities = df["Source City"].str.lower().unique()
    brands = df["Brand"].str.lower().unique()

    detected_city = None
    detected_brand = None

    for city in cities:
        if city and city in q:
            detected_city = city
            break

    for brand in brands:
        if brand and brand in q:
            detected_brand = brand
            break

    return detected_city, detected_brand


# ---------------- EXACT SEARCH ----------------

def exact_search(df, query):
    q = normalize_query(query)

    for _, row in df.iterrows():

        if q in str(row.get("Customer GST No.", "")).lower():
            return row

        if q in str(row.get("Customer account", "")).lower():
            return row

        if q in str(row.get("Name", "")).lower():
            return row

    return None


# ---------------- WEIGHTED SCORE ----------------

def weighted_score(row, query):
    q = normalize_query(query)

    base = fuzz.token_set_ratio(q, row["search_text"])

    bonus = 0
    for field in IMPORTANT_FIELDS:
        value = str(row.get(field, "")).lower()
        if value:
            bonus += fuzz.partial_ratio(q, value) * 0.5

    return base + bonus


# ---------------- UNIQUE VENDORS ----------------

def unique_vendors(df_subset):
    seen = set()
    unique_rows = []

    for _, row in df_subset.iterrows():
        key = str(row.get("Customer account", ""))

        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    return unique_rows


# ---------------- FUZZY SEARCH ----------------

def fuzzy_search(df, query):
    detected_city, detected_brand = extract_filters(df, query)

    filtered_df = df

    # 🔥 Apply city filter first
    if detected_city:
        filtered_df = filtered_df[
            filtered_df["Source City"].str.lower() == detected_city
        ]

    # 🔥 Apply brand filter
    if detected_brand:
        filtered_df = filtered_df[
            filtered_df["Brand"].str.lower() == detected_brand
        ]

    if filtered_df.empty:
        return None

    scores = []

    for i, row in filtered_df.iterrows():
        score = weighted_score(row, query)
        scores.append((score, i))

    scores.sort(reverse=True)

    # 🔥 Strong threshold now
    filtered_scores = [(s, i) for s, i in scores if s > 60]

    if not filtered_scores:
        return None

    top_indices = [i for _, i in filtered_scores[:10]]
    results = filtered_df.loc[top_indices]

    return unique_vendors(results)


# ---------------- FINAL SEARCH ----------------

def search(df, query):
    result = exact_search(df, query)

    if result is not None:
        return result

    return fuzzy_search(df, query)