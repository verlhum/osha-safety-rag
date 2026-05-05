import sys

from src.retrieval.search import IncidentRetriever
from src.rag.generate_answer import generate_answer


def main():
    query = " ".join(sys.argv[1:]) or "What are common forklift hand injury patterns?"

    retriever = IncidentRetriever()

    records = retriever.search(
        query=query,
        k=10,
        candidate_k=1500,
        filters=None,
    )

    print("\nRetrieved records:", len(records))

    answer = generate_answer(query, records)

    print("\n" + "=" * 80)
    print("ANSWER:\n")
    print(answer)
    print("=" * 80)


if __name__ == "__main__":
    main()
