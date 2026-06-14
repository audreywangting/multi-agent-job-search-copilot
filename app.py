import json
import os
import hashlib
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pypdf import PdfReader

load_dotenv()

from rag.chunker import chunk_text
from rag.embeddings import (
    EMBEDDING_MODEL,
    embed_chunks,
)
from rag.retriever import retrieve_top_k


APP_DIR = Path(__file__).parent
PROMPT_PATH = APP_DIR / "prompts" / "match_analysis_prompt.txt"
CHAT_MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 700


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "you",
    "your",
}

GENERIC_JOB_WORDS = STOPWORDS | {
    "ability",
    "about",
    "across",
    "also",
    "apply",
    "based",
    "build",
    "building",
    "candidate",
    "company",
    "create",
    "design",
    "develop",
    "drive",
    "excellent",
    "experience",
    "familiar",
    "help",
    "high",
    "including",
    "integrate",
    "integrating",
    "intern",
    "internship",
    "join",
    "looking",
    "manage",
    "new",
    "preferred",
    "plus",
    "product",
    "projects",
    "required",
    "responsibilities",
    "role",
    "scale",
    "strong",
    "support",
    "team",
    "teams",
    "tools",
    "using",
    "work",
    "working",
}

SKILL_TERMS = {
    "Agentic Workflows": ["agentic workflow", "agentic workflows", "agent workflow"],
    "FastAPI": ["fastapi", "fast api"],
    "Internal Tools": ["internal tool", "internal tools"],
    "LangChain": ["langchain", "lang chain"],
    "LangGraph": ["langgraph", "lang graph"],
    "Notion API": ["notion api", "notion"],
    "OpenAI API": ["openai api", "openai", "gpt"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "Slack API": ["slack api", "slack"],
    "Streamlit": ["streamlit"],
    "Telemetry": ["telemetry", "tracing", "observability", "logging"],
    "API Integration": ["api integration", "api integrations", "apis", "api"],
    "Automation": ["automation", "automations", "automated"],
    "Backend APIs": ["backend api", "backend apis", "backend"],
    "Embeddings": ["embedding", "embeddings", "vector search"],
    "Evaluation": ["evaluation", "evals", "eval"],
    "FAISS": ["faiss"],
    "JavaScript": ["javascript", "typescript"],
    "Python": ["python"],
    "SQL": ["sql"],
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


def build_evidence_text(retrieved_chunks: list[dict]) -> str:
    evidence_blocks = []
    for chunk in retrieved_chunks:
        evidence_blocks.append(
            (
                f"Chunk {chunk['chunk_id']} "
                f"(similarity: {chunk.get('similarity', 0):.3f})\n"
                f"{chunk['text']}"
            )
        )
    return "\n\n---\n\n".join(evidence_blocks)


def important_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9+#]+(?:\.[a-z0-9+#]+)*", text.lower())
    return {
        word
        for word in words
        if len(word) > 2 and word not in GENERIC_JOB_WORDS and not word.isdigit()
    }


def text_has_variant(text: str, variant: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(variant.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def find_skill_terms(text: str) -> list[str]:
    found_terms = []
    for display_name, variants in SKILL_TERMS.items():
        if any(text_has_variant(text, variant) for variant in variants):
            found_terms.append(display_name)
    return found_terms


def format_keyword(word: str) -> str:
    acronyms = {"api", "apis", "llm", "llms", "rag", "sql", "ui", "ux"}
    if word in acronyms:
        return word.upper()
    if word in {"openai"}:
        return "OpenAI"
    return word.replace("+", "+").title()


def recommendation_for_score(score: int) -> str:
    if score >= 75:
        return "Strong Apply"
    if score >= 55:
        return "Apply"
    if score >= 35:
        return "Low Priority"
    return "Skip"


def build_mock_analysis(
    job_description: str,
    retrieved_chunks: list[dict],
) -> dict:
    evidence_text = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    jd_skill_terms = find_skill_terms(job_description)
    evidence_skill_terms = set(find_skill_terms(evidence_text))

    jd_words = important_words(job_description)
    evidence_words = important_words(evidence_text)
    matched_words = sorted(jd_words & evidence_words)
    missing_words = sorted(jd_words - evidence_words)

    matched_skills = [
        skill for skill in jd_skill_terms if skill in evidence_skill_terms
    ]
    matched_skills.extend(format_keyword(word) for word in matched_words)
    matched_skills = list(dict.fromkeys(matched_skills))[:12]

    partial_matches = [
        skill
        for skill in jd_skill_terms
        if skill not in evidence_skill_terms
        and any(word in evidence_words for word in important_words(skill))
    ][:8]

    missing_skills = [
        skill for skill in jd_skill_terms if skill not in evidence_skill_terms
    ]
    missing_skills.extend(format_keyword(word) for word in missing_words)
    missing_skills = [
        skill for skill in dict.fromkeys(missing_skills) if skill not in partial_matches
    ][:12]

    skill_overlap = len(set(jd_skill_terms) & evidence_skill_terms)
    total_signal_count = max(len(set(jd_skill_terms)) + len(jd_words), 1)
    matched_signal_count = skill_overlap + len(jd_words & evidence_words)
    overlap_ratio = matched_signal_count / total_signal_count
    match_score = min(100, round(overlap_ratio * 100))

    if matched_skills:
        improvement_focus = ", ".join(missing_skills[:3]) or "the JD's priority skills"
        resume_improvements = [
            "Move the most relevant project or experience closer to the top of the resume.",
            f"Add clearer evidence for {improvement_focus} if you have that experience.",
            "Mirror the job description's terminology where it accurately describes your work.",
        ]
    else:
        resume_improvements = [
            "Add projects or experience that directly match the job description's core skills.",
            "Use the same terminology as the job description where it is accurate.",
            "Include measurable outcomes for the most relevant work.",
        ]

    reasoning_summary = (
        "Mock analysis used local keyword overlap between the job description and "
        f"retrieved resume evidence. It found {len(matched_skills)} matched signals "
        f"and {len(missing_skills)} missing signals."
    )

    return {
        "match_score": match_score,
        "recommendation": recommendation_for_score(match_score),
        "matched_skills": matched_skills,
        "partial_matches": partial_matches,
        "missing_skills": missing_skills,
        "resume_improvements": resume_improvements,
        "reasoning_summary": reasoning_summary,
    }


def retrieve_top_k_by_keyword_overlap(
    job_description: str,
    chunks: list[dict],
    k: int = 5,
) -> list[dict]:
    job_words = important_words(job_description)
    scored_chunks = []

    for chunk in chunks:
        chunk_words = important_words(chunk["text"])
        overlap_count = len(job_words & chunk_words)
        score = overlap_count / max(len(job_words), 1)
        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "similarity": score,
                "overlap_count": overlap_count,
                "retrieval_method": "keyword_overlap",
            }
        )

    return sorted(
        scored_chunks,
        key=lambda chunk: (chunk["overlap_count"], chunk["similarity"]),
        reverse=True,
    )[:k]


def cache_key_for_resume(resume_text: str, chunk_size: int = 900, overlap: int = 150) -> str:
    raw_key = f"{chunk_size}:{overlap}:{resume_text}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def get_cached_embedded_chunks(
    resume_text: str,
    chunks: list[dict],
    client,
) -> list[dict]:
    cache_key = cache_key_for_resume(resume_text)
    cache = st.session_state.get("resume_embedding_cache")

    if cache and cache.get("key") == cache_key:
        return cache["embedded_chunks"]

    embedded_chunks = embed_chunks(chunks, client)
    st.session_state["resume_embedding_cache"] = {
        "key": cache_key,
        "embedded_chunks": embedded_chunks,
    }
    return embedded_chunks


def analyze_with_openai(
    retrieved_chunks: list[dict],
    job_description: str,
    client,
) -> tuple[dict | None, str]:
    prompt_template = load_prompt_template()
    user_prompt = prompt_template.format(
        retrieved_resume_evidence=build_evidence_text(retrieved_chunks)[:10000],
        job_description=job_description[:8000],
    )

    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a careful job application fit analyst. Return compact, "
                    "valid JSON only. Use only the retrieved resume evidence."
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
    print("Raw model response:")
    print(raw_text)
    print("repr(raw_response):")
    print(repr(raw_text))
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


def render_retrieved_evidence(
    retrieved_chunks: list[dict],
    *,
    retrieval_label: str,
    note: str | None = None,
) -> None:
    st.subheader("Retrieved Resume Evidence")
    if note:
        st.caption(note)

    if not retrieved_chunks:
        st.caption("No retrieved chunks to display.")
        return

    for chunk in retrieved_chunks:
        similarity = chunk.get("similarity")
        similarity_label = (
            f"{similarity:.3f}" if isinstance(similarity, (int, float)) else "N/A"
        )
        label = (
            f"Chunk {chunk.get('chunk_id', 'N/A')} | "
            f"{retrieval_label} {similarity_label}"
        )
        if "overlap_count" in chunk:
            label = f"{label} | {chunk['overlap_count']} keyword overlaps"

        with st.expander(label):
            text = chunk.get("text", "")
            preview = text[:1200] + ("..." if len(text) > 1200 else "")
            st.write(preview)


def render_raw_llm_response(raw_response: str) -> None:
    st.subheader("Raw LLM Response")
    st.code(raw_response, language="text")


def main() -> None:
    st.set_page_config(page_title="Multi-Agent Job Search Copilot")

    st.title("Multi-Agent Job Search Copilot")
    st.caption("Day 2 MVP: resume RAG retrieval plus job-fit analysis.")

    with st.sidebar:
        st.header("Analysis Mode")
        mock_mode = st.toggle(
            "Mock mode",
            value=True,
            help=(
                "On by default. Mock mode makes no OpenAI API calls and does not "
                "incur API costs."
            ),
        )
        top_k = st.slider("Top-K resume chunks", min_value=3, max_value=8, value=5)

        if mock_mode:
            st.success("Mock mode is active. No OpenAI API calls will be made.")
            st.caption("Debug mode: mock")
            st.caption(f"Configured embedding model: {EMBEDDING_MODEL}")
        else:
            st.warning(
                "Real API mode may incur OpenAI API costs because it uses both "
                "embeddings and chat completion when you click Analyze.",
            )
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY is missing. Real API analysis is disabled.")
            st.caption("Debug mode: real")
            st.caption(f"Configured embedding model: {EMBEDDING_MODEL}")
            st.caption(f"Chat model: {CHAT_MODEL}")
            st.caption(f"Max output tokens: {MAX_OUTPUT_TOKENS}")

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
                chunks = chunk_text(resume_text)
                st.caption(f"Resume extracted and split into {len(chunks)} chunks.")
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

        chunks = chunk_text(resume_text)
        if not chunks:
            st.error("The resume text could not be split into chunks.")
            return

        if mock_mode:
            st.info("Mock mode is enabled, so this result did not call the OpenAI API.")
            retrieved_chunks = retrieve_top_k_by_keyword_overlap(
                job_description=job_description.strip(),
                chunks=chunks,
                k=top_k,
            )
            render_retrieved_evidence(
                retrieved_chunks,
                retrieval_label="mock keyword score",
                note=(
                    "Mock retrieval uses local keyword overlap on your uploaded resume, "
                    "not embeddings."
                ),
            )
            render_result(
                build_mock_analysis(
                    job_description=job_description.strip(),
                    retrieved_chunks=retrieved_chunks,
                )
            )
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error(
                "OPENAI_API_KEY is missing. Add it to your environment or switch back "
                "to mock mode."
            )
            return

        with st.spinner("Embedding resume chunks and retrieving relevant evidence..."):
            try:
                client = OpenAI(api_key=api_key)
                embedded_chunks = get_cached_embedded_chunks(
                    resume_text=resume_text,
                    chunks=chunks,
                    client=client,
                )
                retrieved_chunks = retrieve_top_k(
                    job_description=job_description.strip(),
                    embedded_chunks=embedded_chunks,
                    client=client,
                    k=top_k,
                )
                parsed_result, raw_response = analyze_with_openai(
                    retrieved_chunks=retrieved_chunks,
                    job_description=job_description.strip(),
                    client=client,
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
            render_raw_llm_response(raw_response)
            render_retrieved_evidence(
                retrieved_chunks,
                retrieval_label="semantic similarity",
                note="Real retrieval uses OpenAI embeddings and cosine similarity.",
            )
            render_result(parsed_result)
        else:
            render_raw_llm_response(raw_response)
            render_retrieved_evidence(
                retrieved_chunks,
                retrieval_label="semantic similarity",
                note="Real retrieval uses OpenAI embeddings and cosine similarity.",
            )
            st.warning("The model response was not valid JSON. Showing raw response.")
            st.code(raw_response, language="text")


if __name__ == "__main__":
    main()
