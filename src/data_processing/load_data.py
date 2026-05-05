 # src/data_processing/load_data.py

import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    "eventdate",
    "state",
    "primary_naics",
    "final_narrative",
]


def load_osha_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw OSHA dataset and apply minimal normalization.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load file
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    elif file_path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Basic sanity check
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
