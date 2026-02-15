# Intent & Sentiment-Aware Information Retrieval (Financial Q&A)

An **intent-aware** and **sentiment-diverse** information retrieval pipeline for financial Community Question Answering (CQA).  
The system is designed to help **non-expert users** interpret financial questions more safely by:
- returning **high-precision, factual** results for objective queries, and
- surfacing **multiple viewpoints** (via sentiment diversity) for advice-seeking / subjective queries.

> ⚠️ **Disclaimer:** This is a research project for information retrieval. It does **not** provide financial advice.

---

## What this project does

This project implements and evaluates a hybrid retrieval pipeline that combines:

### 1) Sparse retrieval (first stage)
- **BM25** (PyTerrier/Terrier) as the candidate generator.

### 2) Relevance improvements (optional components)
- **Query expansion** (e.g., acronym expansion, semantic expansion, and/or query rewriting).
- **Document augmentation** via **Doc2Query** (generate synthetic queries for documents to reduce vocabulary mismatch).

### 3) Neural re-ranking (second stage)
- A **Cross-Encoder** re-ranker (e.g., MS MARCO MiniLM cross-encoder) to improve semantic relevance.

### 4) Risk-aware diversity re-ranking (for subjective queries)
- **Intent classification** (objective vs advice/opinion-seeking).
- If the query is subjective, apply a **sentiment-aware diversity re-ranker** so results expose **contrasting sentiment/stance** (e.g., risk warnings vs optimistic takes), rather than an “echo chamber”.
- Includes a **sliding-window** approach for long forum threads to avoid missing important content buried deep in the text.

---

## Dataset

The notebook uses the **BEIR FiQA** test split through PyTerrier’s `irds:` integration (via `pt.get_dataset(...)`).  
Documents are forum-style financial discussions with high variance in length and writing style.

---

## Repository contents

- `Financial_Question_Answering.ipynb` — main notebook with:
  - dataset loading
  - indexing
  - baseline experiments (BM25 / PL2 / TF-IDF, etc.)
  - advanced pipelines (Doc2Query + neural re-ranking + intent/sentiment-aware diversity)
  - precomputed expandend documents
  - precomputed expandend queries
- `Project_Report.pdf` — methodology + results write-up

