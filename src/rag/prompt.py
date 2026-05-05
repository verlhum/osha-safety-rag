# src/rag/prompt.py

from typing import Any


def format_incident_context(records: list[dict[str, Any]]) -> str:
    """
    Convert retrieved OSHA incident records into LLM-readable context.
    """

    formatted_records = []

    for i, record in enumerate(records, start=1):
        formatted_records.append(
            f"""
[Retrieved OSHA Record]
Incident ID: {record.get("incident_id")}
Date: {record.get("event_date")}
State: {record.get("state")}
Employer: {record.get("employer")}
Industry Code: {record.get("industry_code")}
Hospitalized: {record.get("hospitalized")}
Amputation: {record.get("amputation")}
Loss of Eye: {record.get("loss_of_eye")}
Injury Nature: {record.get("injury_nature")}
Body Part: {record.get("body_part")}
Event Type: {record.get("event_type")}
Source Type: {record.get("source_type")}
Secondary Source Type: {record.get("secondary_source_type")}
Description: {record.get("incident_description")}
""".strip()
        )

    return "\n\n".join(formatted_records)


def build_rag_prompt(question: str, records: list[dict[str, Any]]) -> str:
    """
    Build a grounded RAG prompt using retrieved OSHA incident records.
    """

    context = format_incident_context(records)

    return f"""
You are an occupational safety analysis assistant.

Answer the user's question using only the OSHA incident records provided below.

Rules:

Grounding:
- Use only the provided OSHA incident records.
- Do not use outside knowledge.
- Every claim must be supported by at least one cited OSHA Incident ID.
- If a claim cannot be supported by a citation, do not include it.

Citations:
- Every bullet must include at least one citation.
- Cite using this exact format: (Incident ID: 2019010532)

Scope:
- Describe only qualitative patterns observed in the retrieved records.
- Focus on specific mechanisms, actions, or conditions that lead to injury (e.g., operator behavior, equipment interaction, environmental conditions).
- Avoid vague or generic summaries.
- The answer reflects only the retrieved records, not the full OSHA dataset.
- If the records are insufficient, state that clearly.
- Avoid repeating similar patterns with different wording; combine overlapping patterns into a single, more general mechanism.
- Distinguish between:
  - the situation or action leading to injury (mechanism)
  - the type of injury (outcome)
- Prefer grouping by mechanism rather than outcome.

Prohibited content:
- Do not provide counts, totals, percentages, rankings, or estimates of frequency.
- Do not use frequency language such as "common", "often", "many", or similar.
- Do not invent causes, categories, or trends not present in the records.
- Do not provide safety recommendations unless explicitly asked.
- Do not list more than one pattern about hands caught/crushed between forklifts and objects; consolidate those into one pattern.

Content constraints:
- You may mention injuries or outcomes (e.g., fatalities) only if explicitly present in the records.
- Only cite Incident IDs that appear exactly in the retrieved records.
- Copy Incident IDs exactly from the records. Do not modify or infer them.

Required answer format:

Based only on the retrieved OSHA records, recurring patterns include:

- Pattern: [specific mechanism or situation leading to injury]
  Evidence: [brief explanation with cited Incident IDs]

- Pattern: [specific mechanism or situation leading to injury]
  Evidence: [brief explanation with cited Incident IDs]

Limitations: These patterns are based only on the retrieved records and are not a statistical summary.

User question:
{question}

Retrieved OSHA incident records:
{context}

Answer:
""".strip()
