 # src/data_processing/build_search_text.py

import pandas as pd


# Replace NA values with empty string
def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_search_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["search_text"] = df.apply(
        lambda row: " ".join(
            part for part in [
                f"Incident description: {_safe_text(row['incident_description'])}",
                f"Injury nature: {_safe_text(row['injury_nature'])}",
                f"Body part: {_safe_text(row['body_part'])}",
                f"Event type: {_safe_text(row['event_type'])}",
                f"Source: {_safe_text(row['source_type'])}",
                f"Secondary source: {_safe_text(row['secondary_source_type'])}",
                f"Employer: {_safe_text(row['employer'])}",
                f"Location: {_safe_text(row['city'])}, {_safe_text(row['state'])}",
                f"Industry code: {_safe_text(row['industry_code'])}",
            ]
            if part.strip()
        ),
        axis=1,
    )

    return df
