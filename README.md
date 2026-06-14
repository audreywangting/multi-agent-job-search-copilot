# Multi-Agent Job Search Copilot

An AI-powered job search assistant that evaluates resume-job fit using retrieval-augmented generation (RAG), semantic resume retrieval, and structured LLM analysis.

The system extracts resume content from uploaded PDFs, retrieves the most relevant resume evidence for a given job description, and generates structured skill-gap analysis, match recommendations, and resume improvement suggestions.

## Architecture

```text
Resume PDF
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Similarity Search
    ↓
Top-K Evidence Retrieval
    ↓
LLM Analysis
    ↓
Structured Recommendation Report
```

## Features

* Upload and analyze resume PDFs
* Retrieval-Augmented Generation (RAG) pipeline
* Semantic resume retrieval using OpenAI embeddings
* Configurable Top-K evidence selection
* Structured job-fit analysis
* Skill-gap identification
* Resume improvement recommendations
* Mock mode for local testing without API usage
* Real API mode for semantic retrieval and LLM reasoning
* Structured JSON output for downstream automation

## Example Output

The application generates:

* Match Score
* Recommendation (Strong Apply / Apply / Low Priority / Skip)
* Matched Skills
* Partial Matches
* Missing Skills
* Suggested Resume Improvements
* Reasoning Summary
* Retrieved Resume Evidence with similarity scores

## Technical Implementation

### Resume Extraction

Uploaded PDF resumes are parsed using `pypdf` and converted into raw text for downstream processing.

### Resume Chunking

The extracted resume text is split into overlapping chunks to improve retrieval quality while maintaining manageable context windows.

Default settings:

* Chunk Size: 900 characters
* Overlap: 150 characters

### Embeddings

In Real API Mode, resume chunks are embedded using:

```text
text-embedding-3-small
```

Embeddings are cached in Streamlit session state to avoid unnecessary recomputation when the uploaded resume remains unchanged.

### Semantic Retrieval

When a job description is submitted:

1. The job description is embedded.
2. Cosine similarity is computed against all resume chunk embeddings.
3. The Top-K most relevant chunks are retrieved.
4. Only the retrieved evidence is passed to the language model.

This approach reduces irrelevant context and improves analysis quality.

### Structured LLM Analysis

Retrieved resume evidence and the target job description are provided to `gpt-4o-mini`.

The model returns structured JSON containing:

```json
{
  "match_score": 85,
  "recommendation": "Strong Apply",
  "matched_skills": [],
  "partial_matches": [],
  "missing_skills": [],
  "resume_improvements": [],
  "reasoning_summary": ""
}
```

The analysis is constrained to retrieved evidence only to reduce unsupported conclusions.

## Retrieval Pipeline

```text
Resume
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Vector Similarity Search
    ↓
Top-K Evidence
    ↓
LLM Analysis
```

## Mock Mode

Mock Mode is enabled by default.

In Mock Mode:

* No OpenAI API calls are made
* Resume extraction and chunking still occur
* Retrieval uses local keyword-overlap scoring
* Results are generated without external API usage

This mode is useful for local development and testing.

## Real API Mode

In Real API Mode:

* Resume chunks are embedded with OpenAI embeddings
* Semantic similarity retrieval is performed
* Retrieved evidence is analyzed using `gpt-4o-mini`

OpenAI API calls occur only when:

1. Mock Mode is disabled
2. The user clicks Analyze

## Cost Considerations

Real API Mode may incur OpenAI API costs because it uses:

* Embedding generation
* Chat completion

The application avoids unnecessary API usage by:

* Caching embeddings within a session
* Limiting context to retrieved evidence
* Only executing analysis on user request

## API Key Safety

API keys are loaded exclusively through environment variables.

The application:

* Does not hardcode API keys
* Does not commit API keys
* Does not expose API keys in logs
* Ignores `.env` through Git

Example:

```env
OPENAI_API_KEY=your_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Configure:

```env
OPENAI_API_KEY=your_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Run the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Project Structure

```text
job-search-copilot/
├── app.py
├── rag/
│   ├── __init__.py
│   ├── chunker.py
│   ├── embeddings.py
│   └── retriever.py
├── prompts/
│   └── match_analysis_prompt.txt
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Future Enhancements

* LangGraph-based multi-agent orchestration
* Dedicated Job Description Parsing Agent
* Resume Retrieval Agent
* Skill Gap Analysis Agent
* Recommendation Generation Agent
* Telemetry and evaluation dashboard
* Retrieval quality benchmarking
* Agent execution tracing
* Resume version comparison
* Application prioritization across multiple job postings
