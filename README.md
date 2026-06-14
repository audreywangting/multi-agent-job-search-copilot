# Multi-Agent Job Search Copilot

A 3-day side project MVP for a future LangGraph-based multi-agent job search assistant. The Day 2 version is a Streamlit app that compares a resume PDF with a pasted internship job description using a small RAG pipeline.

Instead of sending the entire resume to the chat model, the app chunks the resume, embeds the chunks, retrieves the most relevant sections for the job description, and sends only those Top-K sections to the model for analysis.

## MVP Features

- Upload a resume PDF.
- Extract resume text with `pypdf`.
- Chunk the resume with simple character-based chunking.
- Embed resume chunks with `text-embedding-3-small`.
- Retrieve Top-K resume chunks with cosine similarity.
- Paste a job description.
- Analyze resume-job fit from retrieved evidence only.
- Display:
  - match score
  - recommendation: Strong Apply / Apply / Low Priority / Skip
  - matched skills
  - partial matches
  - missing skills
  - suggested resume improvements
  - short reasoning summary
  - retrieved resume evidence with chunk IDs and similarity scores
- Mock/demo mode that uses local keyword-overlap retrieval on the uploaded resume without any OpenAI API call.
- Real API mode that uses OpenAI embeddings and semantic similarity only when mock mode is disabled and the `Analyze` button is clicked.

## How Day 2 RAG Works

### Resume Chunking

The app splits extracted resume text into overlapping character-based chunks. This keeps the implementation simple and avoids adding heavier frameworks before they are needed.

Default chunk settings:

- chunk size: 900 characters
- overlap: 150 characters

### Embeddings

In real API mode, each resume chunk is embedded with `text-embedding-3-small`. The app stores the embedded resume chunks in Streamlit session state so repeated analyses in the same session can reuse the resume embeddings when the resume has not changed.

### Top-K Retrieval

In mock mode, when you click `Analyze`, the app lowercases the job description and each resume chunk, counts overlap of important words, and returns the Top-K actual resume chunks by keyword-overlap score. This is local-only and does not call OpenAI.

In real API mode, when you click `Analyze`, the app embeds the job description, compares it with each embedded resume chunk using cosine similarity, and returns the Top-K chunks by semantic similarity. The Top-K value is configurable in the sidebar from 3 to 8, with a default of 5.

Only the retrieved chunks and the job description are sent to the chat model.

## Cost Warning

Real API mode may incur OpenAI API costs because it uses both embeddings and chat completion. Mock mode is enabled by default and does not call the OpenAI API.

This app does not use background jobs, scheduled calls, scraping, loops, or automatic repeated API calls. OpenAI API calls happen only when mock mode is disabled and you click `Analyze`.

## API Key Safety

Do not hardcode, print, log, commit, or share your OpenAI API key. This project reads the key from an environment variable only.

The `.env` file is ignored by git. Use `.env.example` as a template.

## Setup

```bash
cd job-search-copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your shell uses `python3` instead of `python`, run:

```bash
python3 -m venv .venv
```

## Create `.env`

Copy the example file:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
OPENAI_API_KEY=your_real_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

You can skip this step if you only want to use mock mode.

If your OpenAI project does not have access to the configured embedding model, real mode will show the exact OpenAI error message. Mock mode still works without any OpenAI API access.

## Run Streamlit

```bash
streamlit run app.py
```

Open the local URL Streamlit prints in your terminal.

## Mock Mode

Mock mode is enabled by default in the sidebar. In mock mode, the app extracts and chunks the uploaded resume normally, retrieves actual resume chunks with local keyword-overlap scoring, and generates a simple keyword-based analysis from the job description plus retrieved evidence. It does not call the OpenAI API.

To use the real API, turn off `Mock mode` in the sidebar, make sure `OPENAI_API_KEY` is set, upload a resume PDF, paste a job description, choose Top-K, and click `Analyze`.

## Project Structure

```text
job-search-copilot/
├── app.py
├── rag/
│   ├── __init__.py
│   ├── chunker.py
│   ├── embeddings.py
│   └── retriever.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
└── prompts/
    └── match_analysis_prompt.txt
```

## Next Step

Day 3:
- Build a LangGraph multi-agent workflow.
- Add a JD parser agent.
- Add a resume retrieval agent.
- Add a skill gap agent.
- Add a recommendation agent.
- Add telemetry for debugging, tracing, and evaluation.
