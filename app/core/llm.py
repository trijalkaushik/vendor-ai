import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"


def generate_response(context, query):
    prompt = f"""
You are a strict data extraction assistant.

Rules:
- Use ONLY the provided context
- Extract only relevant fields based on the query
- Do NOT add external knowledge
- Do NOT hallucinate
- If no data found, say: "No relevant vendor found."

Context:
{context}

User Query:
{query}

Answer:
"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })

    return response.json()["response"]