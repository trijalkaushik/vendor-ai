from app.core.loader import load_data
from app.core.search import search
from app.core.llm import generate_response

# Load data once
df = load_data()


def format_context(result):
    def format_row(r):
        parts = []

        # Include ALL columns
        for col, val in r.items():
            if val and str(val).strip() != "":
                parts.append(f"{col}: {val}")

        return " | ".join(parts)

    # Multiple vendors
    if isinstance(result, list):
        return "\n".join([format_row(r) for r in result])

    # Single vendor
    return format_row(result)


def chat(query):
    result = search(df, query)

    if result is None:
        return "No relevant vendor found."

    context = format_context(result)

    return generate_response(context, query)


if __name__ == "__main__":
    print("Vendor AI ready. Type 'exit' to quit.")

    while True:
        q = input("\nAsk: ").strip()

        if not q:
            continue

        if q.lower() == "exit":
            break

        response = chat(q)
        print("\n", response)