 # src/data_processing/clean_data.py

import pandas as pd


COLUMN_RENAMES = {
    "id": "incident_id",
    "eventdate": "event_date",
    "primary_naics": "industry_code",
    "final_narrative": "incident_description",
    "naturetitle": "injury_nature",
    "part_of_body_title": "body_part",
    "eventtitle": "event_type",
    "sourcetitle": "source_type",
    "secondary_source_title": "secondary_source_type",
}


KEEP_COLUMNS = [
    "incident_id",
    "event_date",
    "employer",
    "city",
    "state",
    "zip",
    "latitude",
    "longitude",
    "industry_code",
    "hospitalized",
    "amputation",
    "loss_of_eye",
    "inspection",
    "incident_description",
    "injury_nature",
    "body_part",
    "event_type",
    "source_type",
    "secondary_source_type",
    "federalstate",
]


def clean_osha_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns=COLUMN_RENAMES)
    
    # Verify that expected columns are present
    missing = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after rename: {missing}")

    df = df[KEEP_COLUMNS]

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    text_cols = [
        "employer",
        "city",
        "state",
        "incident_description",
        "injury_nature",
        "body_part",
        "event_type",
        "source_type",
        "secondary_source_type",
    ]

    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
    
    # Drop if date or description are missing
    df = df.dropna(subset=["event_date", "incident_description"])
    df = df[df["incident_description"].str.len() > 0]

    return df.reset_index(drop=True)
