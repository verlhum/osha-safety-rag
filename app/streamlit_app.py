# app/streamlit_app.py

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval.search import IncidentRetriever
from src.rag.generate_answer import generate_answer

import re

def make_anchor_id(incident_id) -> str:
    return f"incident-{incident_id}"

def normalize_citations(text: str) -> str:
    # Existing parenthetical handler
    def replace_parenthetical(match):
        content = match.group(1)

        if "Incident ID" not in content:
            return match.group(0)

        ids = re.findall(r"\d{7,}", content)

        if not ids:
            return match.group(0)

        return " ".join(f"(Incident ID: {incident_id})" for incident_id in ids)

    text = re.sub(r"\(([^)]*)\)", replace_parenthetical, text)

    # New: handles "Incident IDs 123, 456, and 789"
    def replace_inline_list(match):
        ids_text = match.group(1)
        ids = re.findall(r"\d{7,}", ids_text)

        if not ids:
            return match.group(0)

        return " ".join(f"(Incident ID: {incident_id})" for incident_id in ids)

    text = re.sub(
        r"Incident IDs?\s+((?:\d{7,}[\s,]*(?:and\s*)?)+)",
        replace_inline_list,
        text,
    )

    return text

def link_incident_citations(answer: str, available_ids: set[str]) -> str:
    """
    Convert (Incident ID: 123) into clickable markdown links
    if the ID exists in retrieved records.
    """
    def repl(match):
        incident_id = match.group(1)

        if incident_id in available_ids:
            return f"[(Incident ID: {incident_id})](#{make_anchor_id(incident_id)})"

        return match.group(0)

    return re.sub(r"\(Incident ID:\s*(\d+)\)", repl, answer)

def highlight_ids(text: str) -> str:
    return re.sub(
        r"\(Incident ID: (\d+)\)",
        r"**(Incident ID: \1)**",
        text,
    )

def find_unsupported_citations(answer: str, available_ids: set[str]) -> set[str]:
    cited_ids = set(re.findall(r"Incident ID:\s*(\d+)", answer))
    return cited_ids - available_ids

st.set_page_config(
    page_title="OSHA Safety RAG",
    layout="wide",
)

st.title("OSHA Safety Incident Explorer")
st.caption(
    "Semantic search + grounded LLM summaries over OSHA severe injury records. "
    "Generated answers should be reviewed alongside the retrieved source records."
)

show_raw = st.checkbox("Show raw retrieved records", value=True)

@st.cache_resource
def get_retriever():
    return IncidentRetriever()


retriever = get_retriever()


# Sidebar
st.sidebar.header("Filters")

state = st.sidebar.text_input("State", placeholder="e.g. TEXAS").strip().upper()

amputation_only = st.sidebar.checkbox("Amputation only")
hospitalized_only = st.sidebar.checkbox("Hospitalized only")
loss_of_eye_only = st.sidebar.checkbox("Loss of eye only")

st.sidebar.divider()

use_date_filter = st.sidebar.checkbox("Use date filter")

start_date = None
end_date = None

if use_date_filter:
    start_date = st.sidebar.date_input("Start date", value=date(2015, 1, 1))
    end_date = st.sidebar.date_input("End date", value=date.today())

st.sidebar.divider()

k_display = st.sidebar.slider("Retrieved records to show", 5, 50, 25, step=5)
k_llm = st.sidebar.slider("Records sent to LLM", 5, 20, 10, step=1)
candidate_k = st.sidebar.slider("Candidate search pool", 100, 3000, 1000, step=100)
   

# Main input
query = st.text_input(
    "Ask a safety question",
    value="Describe qualitative patterns in forklift hand injury records. Do not count incidents.",
)

search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Enter a question first.")
        st.stop()

    filters = {
        "state": state if state else None,
        "amputation_only": amputation_only,
        "hospitalized_only": hospitalized_only,
        "loss_of_eye_only": loss_of_eye_only,
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None,
    }

    with st.spinner("Retrieving relevant records..."):
        records = retriever.search(
            query=query,
            k=k_display,
            candidate_k=candidate_k,
            filters=filters,
        )

    if not records:
        st.warning("No matching incidents found. Try broadening the filters or increasing the candidate pool.")
        st.stop()

    records_for_answer = records[: min(k_llm, len(records))]

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieved records", len(records))
    col2.metric("Records sent to LLM", len(records_for_answer))
    col3.metric("Candidate pool", candidate_k)

    st.subheader("Generated Answer")

    with st.spinner("Generating grounded answer..."):
        answer = generate_answer(query, records_for_answer)

    answer = highlight_ids(answer)
    
    available_ids = {str(r.get("incident_id")) for r in records}
    
    answer = normalize_citations(answer)
    
    unsupported_ids = find_unsupported_citations(answer, available_ids)
    
    if unsupported_ids:
        st.warning(
            "The model cited Incident IDs that were not found in the retrieved records. "
            "This may indicate a hallucinated or altered ID.\n\n"
            + ", ".join(sorted(unsupported_ids))
        )
    
    linked_answer = link_incident_citations(answer, available_ids)

    st.markdown(linked_answer)

    st.info(
        "This answer is generated from the retrieved records shown below. "
        "It is not a full statistical analysis of the OSHA dataset."
    )
    
    if show_raw:
        st.subheader("Retrieved Source Records")

        results_df = pd.DataFrame(
            [
                {
                    "score": round(r.get("score", 0), 4),
                    "incident_id": r.get("incident_id"),
                    "event_date": r.get("event_date"),
                    "state": r.get("state"),
                    "employer": r.get("employer"),
                    "injury_nature": r.get("injury_nature"),
                    "body_part": r.get("body_part"),
                    "event_type": r.get("event_type"),
                    "source_type": r.get("source_type"),
                }
                for r in records
            ]
        )

        st.dataframe(results_df, width='stretch')

        st.subheader("Incident Details")

        for record in records:
            incident_id = str(record.get("incident_id"))

            st.markdown(
                f"""
                <div id="{make_anchor_id(incident_id)}" style="position: relative; top: -80px;"></div>
                """,
                unsafe_allow_html=True,
            )
            
            title = (
                f"Score {record.get('score', 0):.4f} | "
                f"Incident ID: {record.get('incident_id')} | "
                f"{record.get('event_date')} | "
                f"{record.get('state')}"
            )

            with st.expander(title):
                st.markdown(f"**Employer:** {record.get('employer')}")
                st.markdown(f"**Industry Code:** {record.get('industry_code')}")
                st.markdown(f"**Hospitalized:** {record.get('hospitalized')}")
                st.markdown(f"**Amputation:** {record.get('amputation')}")
                st.markdown(f"**Loss of Eye:** {record.get('loss_of_eye')}")
                st.markdown(f"**Injury:** {record.get('injury_nature')}")
                st.markdown(f"**Body Part:** {record.get('body_part')}")
                st.markdown(f"**Event Type:** {record.get('event_type')}")
                st.markdown(f"**Source Type:** {record.get('source_type')}")
                st.markdown("**Description:**")
                st.write(record.get("incident_description"))
