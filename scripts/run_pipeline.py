from src.data_processing.load_data import load_osha_data
from src.data_processing.clean_data import clean_osha_data
from src.data_processing.build_search_text import build_search_text


INPUT_PATH = "data/raw/osha_severe_injuries.csv"
OUTPUT_PATH = "data/processed/osha_cleaned.parquet"


def main():
    df = load_osha_data(INPUT_PATH)
    df = clean_osha_data(df)
    df = build_search_text(df)

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved processed data to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main() 
