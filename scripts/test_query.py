import sys

from src.retrieval.search import IncidentRetriever


def print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}\n")

    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Incident ID: {result['incident_id']}")
        print(f"Date: {result['event_date']}")
        print(f"State: {result['state']}")
        print(f"Employer: {result['employer']}")
        print(f"Industry code: {result['industry_code']}")
        print(f"Hospitalized: {result['hospitalized']}")
        print(f"Amputation: {result['amputation']}")
        print(f"Loss of eye: {result['loss_of_eye']}")
        print(f"Injury: {result['injury_nature']}")
        print(f"Body part: {result['body_part']}")
        print(f"Event type: {result['event_type']}")
        print(f"Source: {result['source_type']}")
        print()
        print(result["incident_description"])
        print()


def main():
    query = " ".join(sys.argv[1:]) or "forklift hand injury"

    retriever = IncidentRetriever()

    results = retriever.search(
        query=query,
        k=5,
        candidate_k=500,
        filters={
            "state": "TEXAS",
            "amputation_only": True,
            "start_date": "2020-01-01",
            "end_date": "2021-12-31",
        },
    )

    print_results(query, results)


if __name__ == "__main__":
    main()
