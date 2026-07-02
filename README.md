# RAGProbe

> Evaluate, trace, and compare RAG pipelines. Because "it seems to work" isn't good enough.

---

## What is this?

Most RAG projects stop at "the answer looks right." RAGProbe goes further — it systematically measures **faithfulness**, **answer relevancy**, **context recall**, and **context precision** across multiple LLM versions, logs every run for full trace visibility, and surfaces everything on a live dashboard.

Built for engineers who want to understand their pipeline, not just demo it.

---

## Features

- **Automated evaluation** using RAGAS across 4 core metrics
- **Multi-model comparison** — run GPT-4o vs LLaMA3 side by side and see the difference in numbers
- **LangSmith tracing** — every query logs latency, token count, retrieval steps, and LLM calls
- **Live dashboard** — filter by model, date, or metric; track scores over time
- **REST API** — evaluate queries programmatically via FastAPI endpoints
- **Persistent logging** — every eval run stored in PostgreSQL for historical analysis

---

## Metrics Tracked

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Does the answer actually address the question? |
| Context Recall | Did retrieval surface the right chunks? |
| Context Precision | Were the retrieved chunks actually useful? |

---

## Tech Stack

| Layer | Tool |
|---|---|
| RAG pipeline | LangChain, FAISS |
| LLMs | GPT-4o, LLaMA3 70B |
| Evaluation | RAGAS |
| Observability | LangSmith |
| Backend | FastAPI |
| Database | PostgreSQL |
| Dashboard | Streamlit |
| Deployment | Docker, GCP Cloud Run |

---

## Architecture

```
User Query
    ↓
RAG Pipeline (LangChain + FAISS + LLM)
    ↓
Response + Retrieved Contexts
    ↓
RAGAS Evaluator → faithfulness, relevancy, recall, precision
    ↓
PostgreSQL (stores all runs + scores)
    ↓
FastAPI ←→ Streamlit Dashboard
    +
LangSmith (parallel trace logging)
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/deepikaurikiti/ragprobe.git
cd ragprobe
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Fill in:

```
OPENAI_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=ragprobe
DATABASE_URL=postgresql://user:password@localhost:5432/ragprobe
```

### 3. Run with Docker

```bash
docker-compose up --build
```

### 4. Open the dashboard

```
http://localhost:8501
```

---

## API Endpoints

```
POST /evaluate         — run a query through RAG + RAGAS, log to DB
GET  /runs             — return all eval runs with scores
GET  /compare          — side-by-side model comparison (?model_a=gpt-4o&model_b=llama3)
```

---

## Results

| Model | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|
| GPT-4o | 0.91 | 0.87 | 0.83 |
| LLaMA3 70B | 0.86 | 0.82 | 0.79 |

*Results on a legal document QA dataset. Numbers will vary by domain and chunk strategy.*

---

## Project Structure

```
ragprobe/
├── pipeline/          # LangChain RAG pipeline
├── evaluation/        # RAGAS integration + scoring logic
├── api/               # FastAPI backend
├── dashboard/         # Streamlit app
├── db/                # PostgreSQL schema + migrations
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Why I built this

Most RAG tutorials show you how to get an answer. Very few show you how to know if that answer is any good. This project is my attempt to close that gap — building the kind of eval infrastructure that production AI teams actually use.

---

## Author

**Deepika Urikiti** — [LinkedIn](https://linkedin.com/in/deepikaurikiti) · [Portfolio](https://mellow-lokum-e1c362.netlify.app/) · [GitHub](https://github.com/deepikaurikiti)
