 # src/rag/generate_answer.py

import requests
import re

from src.rag.prompt import build_rag_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"


    
def enforce_citation_format(text: str) -> str:
    # Convert raw numbers like (1234567890) → (Incident ID: 1234567890)
    return re.sub(
        r"\((\d{7,})\)",
        r"(Incident ID: \1)",
        text
    )
    
def normalize_multi_ids(text: str) -> str:
    def repl(match):
        ids = match.group(1).split(",")
        ids = [i.strip() for i in ids]
        return " ".join(f"(Incident ID: {i})" for i in ids)

    return re.sub(
        r"\(Incident ID:\s*([\d,\s]+)\)",
        repl,
        text
    )

def generate_answer(
    question: str,
    records: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a grounded answer using retrieved OSHA records.
    """

    prompt = build_rag_prompt(question, records)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
    )

    response.raise_for_status()

    result = response.json()
    
    answer = result["response"].strip()
    answer = enforce_citation_format(answer)
    answer = normalize_multi_ids(answer)

    return answer
