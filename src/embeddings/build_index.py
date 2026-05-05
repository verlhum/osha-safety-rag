 # src/embeddings/build_index.py

from pathlib import Path
import json

import faiss
import pandas as pd

from src.embeddings.embedder import Embedder


def build_faiss_index(
    processed_data_path: str,
    index_output_path: str,
    metadata_output_path: str,
    text_column: str = "search_text",
) -> None:
    """
    Build a FAISS vector index from processed OSHA data.
    """

    processed_data_path = Path(processed_data_path)
    index_output_path = Path(index_output_path)
    metadata_output_path = Path(metadata_output_path)

    df = pd.read_parquet(processed_data_path)

    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    df = df[df[text_column].notna()].reset_index(drop=True)

    texts = df[text_column].astype(str).tolist()

    embedder = Embedder()
    embeddings = embedder.embed_texts(texts)

    dim = embeddings.shape[1]

    # Inner product works like cosine similarity because embeddings are normalized.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_output_path))

    metadata = df.to_dict(orient="records")

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, default=str)

    print(f"Built FAISS index with {index.ntotal} records")
    print(f"Saved index to {index_output_path}")
    print(f"Saved metadata to {metadata_output_path}")
