import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pypdf import PdfReader


load_dotenv()

APP_DIR = Path(__file__).parent
PROMPT_PATH = APP_DIR / "prompts" / "match_analysis_prompt.txt"
LOW_COST_MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 700


MOCK_RESULT = {
    "match_score": 82,
    "recommendation": "Strong Apply",
    "matched_skills": ["Python", "SQL", "Backend APIs"],
    "partial_matches": ["AWS", "Distributed Systems"],
    "missing_skills": ["Docker", "Kubernetes"],
    "resume_improvements": [
        "Move backend project experience higher on the resume.",
        "Add deployment or cloud-related details if available.",
        "Emphasize system design and API experience.",
    ],
    "reasoning_summary": (
        "The candidate has strong Python, SQL, and backend experience, "
        "but cloud deployment evidence is limited."
    ),
}


def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        page_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            page_text.append(text)
        return "\n\n".join(page_text).strip()
    except Exception as exc:
        raise ValueError(f"Could not extract text from the PDF: {exc}") from exc


def load_prompt_template() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "Analyze the resume against the job description and return only valid JSON "
            "with match_score, recommendation, matched_skills, partial_matches, "
            "missing_skills, resume_improvements, and reasoning_summary."
        )


def analyze_with_openai(resume_text: str, job_description: str) -> tuple[dict | None, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your environment or use mock mode."
        )

    client = OpenAI(api_key=api_key)
    prompt_template = load_prompt_template()
    user_prompt = prompt_template.format(
        resume_text=resume_text[:12000],
        job_description=job_description[:8000],
    )

    response = client.responses.create(
        model=LOW_COST_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a careful job application fit analyst. Return compact, "
                    "valid JSON only. Do not invent experience that is not supported "
                    "by the resume."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={
            "format": {
                "type": "json_schema",
                "name": "job_match_analysis",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "match_score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "recommendation": {
                            "type": "string",
                            "enum": [
                                "Strong Apply",
                                "Apply",
                                "Low Priority",
                                "Skip",
                            ],
                        },
                        "matched_skills": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "partial_matches": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "missing_skills": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "resume_improvements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reasoning_summary": {"type": "string"},
                    },
                    "required": [
                        "match_score",
                        "recommendation",
                        "matched_skills",
                        "partial_matches",
                        "missing_skills",
                        "resume_improvements",
                        "reasoning_summary",
                    ],
                },
                "strict": True,
            }
        },
    )

    raw_text = response.output_text.strip()
    try:
        return json.loads(raw_text), raw_text
    except json.JSONDecodeError:
        return None, raw_text


def render_list(title: str, items: list[str]) -> None:
    st.subheader(title)
    if items:
        for item in items:
            st.write(f"- {item}")
    else:
        st.caption("No items returned.")


def render_result(result: dict) -> None:
    score = result.get("match_score", "N/A")
    recommendation = result.get("recommendation", "N/A")

    metric_col, recommendation_col = st.columns(2)
    metric_col.metric("Match Score", f"{score}/100" if isinstance(score, int) else score)
    recommendation_col.metric("Recommendation", recommendation)

    render_list("Matched Skills", result.get("matched_skills", []))
    render_list("Partial Matches", result.get("partial_matches", []))
    render_list("Missing Skills", result.get("missing_skills", []))
    render_list("Suggested Resume Improvements", result.get("resume_improvements", []))

    st.subheader("Reasoning Summary")
    st.write(result.get("reasoning_summary", "No summary returned."))


def main() -> None:
    st.set_page_config(page_title="Multi-Agent Job Search Copilot")

    st.title("Multi-Agent Job Search Copilot")
    st.caption("Day 1 MVP: resume PDF extraction and job-fit analysis.")

    with st.sidebar:
        st.header("Analysis Mode")
        use_real_api = st.toggle(
            "Enable real OpenAI API mode",
            value=False,
            help=(
                "Off by default. Mock mode makes no API calls and does not incur API costs."
            ),
        )

        if use_real_api:
            st.warning(
                "Real API mode may incur OpenAI API costs when you click Analyze.",
            )
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY is missing. Real API analysis is disabled.")
            st.caption(f"Model: {LOW_COST_MODEL}")
            st.caption(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
        else:
            st.success("Mock mode is active. No OpenAI API calls will be made.")

    uploaded_resume = st.file_uploader("Upload resume PDF", type=["pdf"])
    job_description = st.text_area(
        "Paste job description",
        height=260,
        placeholder="Paste the internship job description here...",
    )

    resume_text = ""
    if uploaded_resume is not None:
        try:
            resume_text = extract_pdf_text(uploaded_resume)
            if resume_text:
                with st.expander("Extracted Resume Text Preview"):
                    st.text_area(
                        "Resume text",
                        value=resume_text[:4000],
                        height=220,
                        disabled=True,
                    )
            else:
                st.warning("No text could be extracted from this PDF.")
        except ValueError as exc:
            st.error(str(exc))

    if st.button("Analyze", type="primary"):
        if not resume_text:
            st.error("Please upload a resume PDF with extractable text before analyzing.")
            return

        if not job_description.strip():
            st.error("Please paste a job description before analyzing.")
            return

        if not use_real_api:
            st.info("Mock mode is enabled, so this result did not call the OpenAI API.")
            render_result(MOCK_RESULT)
            return

        if not os.getenv("OPENAI_API_KEY"):
            st.error(
                "OPENAI_API_KEY is missing. Add it to your environment or switch back "
                "to mock mode."
            )
            return

        with st.spinner("Analyzing resume-job fit..."):
            try:
                parsed_result, raw_response = analyze_with_openai(
                    resume_text=resume_text,
                    job_description=job_description.strip(),
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return
            except OpenAIError as exc:
                st.error(f"OpenAI API error: {exc}")
                return
            except Exception as exc:
                st.error(f"Unexpected analysis error: {exc}")
                return

        if parsed_result is not None:
            render_result(parsed_result)
        else:
            st.warning("The model response was not valid JSON. Showing raw response.")
            st.code(raw_response, language="text")


if __name__ == "__main__":
    main()
