from app.core.loader import load_data
from app.core.search import search, search_raw, extract_intent, generate_answer

# ── Load data once ────────────────────────────────────────────────────────────
df = load_data()

FOLLOW_UP_TRIGGERS = ["same", "that", "their", "its", "those", "these", "them"]


# ── Session class ─────────────────────────────────────────────────────────────
class VendorChatSession:
    def __init__(self, dataframe):
        self.df = dataframe
        self.last_candidates = None
        self.last_query = None

    def _is_followup(self, query: str) -> bool:
        words = query.lower().split()
        return any(trigger in words for trigger in FOLLOW_UP_TRIGGERS)

    def chat(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""

        if self._is_followup(query) and self.last_candidates is not None:
            # Reuse previous search results, just ask a new question about them
            print(f"[chat] Follow-up detected, reusing {len(self.last_candidates)} previous results")
            intent = extract_intent(query)
            return generate_answer(query, self.last_candidates, intent)
        else:
            # Fresh query — run full pipeline
            self.last_candidates = search_raw(self.df, query)
            self.last_query = query
            return search(self.df, query)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check Ollama is running before starting
    try:
        import ollama
        from app.core.search import MODEL
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1, "temperature": 0},
        )
    except Exception as e:
        print(f"[ERROR] Ollama not available or model not pulled: {e}")
        print(f"Run: ollama pull {MODEL}")
        exit(1)

    session = VendorChatSession(df)
    print("Vendor AI ready. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        response = session.chat(q)
        print(f"\n{response}\n")