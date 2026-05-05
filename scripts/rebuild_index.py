# scripts/rebuild_index.py

from src.embeddings.build_index import build_faiss_index


def main():
    build_faiss_index(
        processed_data_path="data/processed/osha_cleaned.parquet",
        index_output_path="data/embeddings/osha_faiss.index",
        metadata_output_path="data/embeddings/osha_metadata.json",
    )


if __name__ == "__main__":
    main()
