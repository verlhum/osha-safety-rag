 import json
from pathlib import Path

import faiss
import pandas as pd

from src.data_processing.load_data import load_osha_data
from src.data_processing.clean_data import clean_osha_data
from src.data_processing.build_search_text import build_search_text
from src.embeddings.embedder import Embedder


RAW_NEW_DATA_PATH = "data/raw/osha_severe_injuries_latest.csv"
PROCESSED_DATA_PATH = "data/processed/osha_cleaned.parquet"
INDEX_PATH = "data/embeddings/osha_faiss.index"
METADATA_PATH = "data/embeddings/osha_metadata.json"


def main():
    processed_path = Path(PROCESSED_DATA_PATH)
    index_path = Path(INDEX_PATH)
    metadata_path = Path(METADATA_PATH)

    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    # Load existing processed records
    existing_df = pd.read_parquet(processed_path)
    existing_ids = set(existing_df["incident_id"].astype(str))

    # Load and clean latest raw OSHA data
    latest_df = load_osha_data(RAW_NEW_DATA_PATH)
    latest_df = clean_osha_data(latest_df)
    latest_df = build_search_text(latest_df)

    latest_df["incident_id"] = latest_df["incident_id"].astype(str)

    # Keep only new records
    new_df = latest_df[~latest_df["incident_id"].isin(existing_ids)].copy()
    new_df = new_df.reset_index(drop=True)

    print(f"Existing records: {len(existing_df)}")
    print(f"Latest records: {len(latest_df)}")
    print(f"New records found: {len(new_df)}")

    if new_df.empty:
        print("No new records to append.")
        return

    # Embed new records
    embedder = Embedder()
    new_embeddings = embedder.embed_texts(new_df["search_text"].astype(str).tolist())

    # Load and update FAISS index
    index = faiss.read_index(str(index_path))
    index.add(new_embeddings)
    faiss.write_index(index, str(index_path))

    # Append processed parquet
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    updated_df.to_parquet(processed_path, index=False)

    # Append metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata.extend(new_df.to_dict(orient="records"))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, default=str)

    print(f"Updated processed records: {len(updated_df)}")
    print(f"Updated FAISS index count: {index.ntotal}")
    print("Incremental append complete.")


if __name__ == "__main__":
    main()
