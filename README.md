# Multi-Agent Job Search Copilot

A 3-day side project MVP for a future LangGraph-based multi-agent job search assistant. Today’s version is a simple Streamlit app that compares a resume PDF with a pasted internship job description and returns a structured fit analysis.

## MVP Features

- Upload a resume PDF.
- Extract resume text with `pypdf`.
- Paste a job description.
- Analyze resume-job fit.
- Display:
  - match score
  - recommendation: Strong Apply / Apply / Low Priority / Skip
  - matched skills
  - partial matches
  - missing skills
  - suggested resume improvements
  - short reasoning summary
- Mock/demo mode that runs without an OpenAI API call.
- Real API mode that only runs when explicitly enabled and the `Analyze` button is clicked.

## Cost Warning

Running analysis in real API mode may incur OpenAI API costs. Mock mode is enabled by default and does not call the OpenAI API.

This app does not use background jobs, scheduled calls, scraping, loops, or automatic repeated API calls. The OpenAI API is called only when real API mode is enabled and you click `Analyze`.

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

## Create `.env`

Copy the example file:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
OPENAI_API_KEY=your_real_api_key_here
```

You can skip this step if you only want to use mock mode.

## Run Streamlit

```bash
streamlit run app.py
```

Open the local URL Streamlit prints in your terminal.

## Mock Mode

Mock mode is enabled by default in the sidebar. In mock mode, the app returns a realistic sample analysis and does not call the OpenAI API.

To use the real API, explicitly enable `Enable real OpenAI API mode` in the sidebar, make sure `OPENAI_API_KEY` is set, upload a resume PDF, paste a job description, and click `Analyze`.

## Next Steps

Day 2:
- Add RAG with resume chunking.
- Generate embeddings.
- Store and retrieve chunks with FAISS.
- Use retrieved resume sections for more grounded analysis.

Day 3:
- Build a LangGraph multi-agent workflow.
- Add specialized agents for resume retrieval, job analysis, skill-gap analysis, ranking, and recommendation writing.
- Add telemetry for debugging and evaluation.
