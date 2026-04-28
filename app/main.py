from app.core.loader import load_data
from app.core.search import search
from app.core.llm import generate_response

last_result = None

# Load data once
df = load_data()


def format_context(result, query):
    def format_row(r):
        q = query.lower()

        parts = [
            f"Name: {r['Name']}",
            f"State: {r.get('State Name','')}",
            f"Account: {r.get('Customer account','')}"
        ]

        if "item" in q or "product" in q:
            parts.append(f"Item: {r.get('Item Name','')}")
            parts.append(f"Brand: {r.get('Brand','')}")

        if "price" in q or "mrp" in q:
            parts.append(f"MRP: {r.get('MRP','')}")
            parts.append(f"Unit Price: {r.get('Unit Price','')}")

        if "sales" in q:
            parts.append(f"Salesman: {r.get('Salesman','')}")
            parts.append(f"ASM: {r.get('ASM','')}")
            
        if "GST" in q or "gst" in q:
            parts.append(f"GST: {r.get('Customer GST No.','')}")
            
        if len(parts) < 4:
            parts.append(f"Segment: {r.get('Product Segment','')}")

        return " | ".join(parts)

    if isinstance(result, list):
        return "\n".join([format_row(r) for r in result])

    return format_row(result)

def is_simple_query(query):
    q = query.lower()
    return any(x in q for x in ["account", "vendor", "name"])


def chat(query):
    global last_result

    q = query.lower()

    # 👉 Handle follow-up queries
    if any(x in q for x in ["same", "that", "their", "its"]):
        if last_result is None:
            return "No previous context available."
        result = last_result
    else:
        result = search(df, query)
        last_result = result  # store for next query

    if result is None:
        return "No relevant vendor found."

    context = format_context(result, query)

    if is_simple_query(query):
        return context

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