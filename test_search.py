from app.core.loader import load_data
from app.core.search import search

df = load_data()

queries = [
    "Kinley",
    "Sprite Delhi",
    "Ranchi vendor",
    "GST",
    "random nonsense"
]

for q in queries:
    print("\n========================")
    print("Query:", q)

    result = search(df, q)

    if result is None:
        print("No relevant results")

    elif isinstance(result, list):
        for r in result:
            print(f"{r['Name']} | {r['Source City']} | {r['Customer GST No.']}")

    else:
        print(result)