import json
from pathlib import Path
from datetime import datetime

import faiss

from src.embeddings.embedder import Embedder


class IncidentRetriever:
    def __init__(
        self,
        index_path: str = "data/embeddings/osha_faiss.index",
        metadata_path: str = "data/embeddings/osha_metadata.json",
    ):
        self.index = faiss.read_index(str(Path(index_path)))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.embedder = Embedder()

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 100,
        filters: dict | None = None,
    ) -> list[dict]:
        query_embedding = self.embedder.embed_query(query)

        candidate_k = max(candidate_k, k)
        scores, indexes = self.index.search(query_embedding, candidate_k)

        results = []

        for score, idx in zip(scores[0], indexes[0]):
            if idx == -1:
                continue

            record = self.metadata[idx]

            if filters and not self._passes_filters(record, filters):
                continue

            results.append(
                {
                    "score": float(score),
                    "incident_id": record.get("incident_id"),
                    "event_date": record.get("event_date"),
                    "state": record.get("state"),
                    "employer": record.get("employer"),
                    "industry_code": record.get("industry_code"),
                    "hospitalized": record.get("hospitalized"),
                    "amputation": record.get("amputation"),
                    "loss_of_eye": record.get("loss_of_eye"),
                    "incident_description": record.get("incident_description"),
                    "injury_nature": record.get("injury_nature"),
                    "body_part": record.get("body_part"),
                    "event_type": record.get("event_type"),
                    "source_type": record.get("source_type"),
                    "secondary_source_type": record.get("secondary_source_type"),
                }
            )

            if len(results) >= k:
                break

        return results

    def _passes_filters(self, record: dict, filters: dict) -> bool:
        if state := filters.get("state"):
            if str(record.get("state", "")).upper() != str(state).upper():
                return False

        if industry_code := filters.get("industry_code"):
            if str(record.get("industry_code", "")) != str(industry_code):
                return False

        if injury_nature := filters.get("injury_nature"):
            if str(record.get("injury_nature", "")).lower() != str(injury_nature).lower():
                return False

        if body_part := filters.get("body_part"):
            if str(record.get("body_part", "")).lower() != str(body_part).lower():
                return False

        if event_type := filters.get("event_type"):
            if str(record.get("event_type", "")).lower() != str(event_type).lower():
                return False

        if source_type := filters.get("source_type"):
            if str(record.get("source_type", "")).lower() != str(source_type).lower():
                return False

        if filters.get("amputation_only"):
            if int(record.get("amputation", 0)) != 1:
                return False

        if filters.get("hospitalized_only"):
            if int(record.get("hospitalized", 0)) != 1:
                return False

        if filters.get("loss_of_eye_only"):
            if int(record.get("loss_of_eye", 0)) != 1:
                return False

        start_date = filters.get("start_date")
        end_date = filters.get("end_date")

        if start_date or end_date:
            record_date = self._parse_date(record.get("event_date"))

            if record_date is None:
                return False

            if start_date and record_date < self._parse_date(start_date):
                return False

            if end_date and record_date > self._parse_date(end_date):
                return False

        return True

    @staticmethod
    def _parse_date(value):
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        value = str(value)

        # Handles strings like "2019-01-16 00:00:00"
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass

        # Handles strings like "1/16/2019"
        try:
            return datetime.strptime(value, "%m/%d/%Y")
        except ValueError:
            return None
