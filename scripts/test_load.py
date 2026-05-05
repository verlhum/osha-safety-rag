from src.data_processing.load_data import load_osha_data
from src.data_processing.clean_data import clean_osha_data
from src.data_processing.build_search_text import build_search_text

df = load_osha_data("data/raw/osha_severe_injuries.csv")
df = clean_osha_data(df)
df = build_search_text(df)

print(df.shape)
print(df[["incident_id", "event_date", "state", "search_text"]].head())
print(df["search_text"].iloc[0])
