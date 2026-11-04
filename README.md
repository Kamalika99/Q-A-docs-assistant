# Solstice Docs Assistant

A documentation Q&A assistant for Solstice Learning. It retrieves supporting sentences from documentation, cites them, and refuses to guess when the documentation does not contain the answer. No LLM, embeddings, or external knowledge are used — only TF-IDF, Snowball stemming, and answerability decision rules.

## Project Structure
```text
solstice-docs-assistant/
│── config.py            # Central thresholds (T1-T4), system constants, messages & typed models
│── nlp.py               # Text cleaning, normalization, stemming, synonym mapping & sentence splitting
│── security.py          # Input validation, regex safeguards & prompt-override detection
│── index.py             # Document loader, schema validation, passage/document-level indexing
│── engine.py            # Retrieval (dual doc + passage score aggregation), evidence selection & gates
│── ui.py                # Streamlit UI & interactive components
│── app.py               # Streamlit application entry point & facade
│── docs.json            # Documentation corpus (10 verified articles)
│── test_app.py          # Pytest suite & evaluation benchmark (31 labelled questions + traps)
│── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Key Architecture & Design

```mermaid
flowchart LR
  Q[Question] --> V[Validate + security filter]
  V -->|refuse| R[INVALID / RESTRICTED]
  V --> A[Normalise + stem]
  A --> S1[Stage 1: Dual TF-IDF ranks all articles<br/>Doc + Passage score aggregation]
  S1 --> S2[Stage 2: Best 1-2 sentences as evidence]
  S2 --> G{T1 score, T2 coverage,<br/>T3 key terms, T4 ambiguity}
  G -->|all pass| ANS[ANSWER + citation]
  G -->|ambiguous / near miss| P[PARTIAL / AMBIGUOUS]
  G -->|no support| AB[ABSTAIN]
```

### Improvements Over Baseline:
1. **Modular Architecture & Separation of Concerns**:
   - Decoupled into single-responsibility modules (`config`, `nlp`, `security`, `index`, `engine`, `ui`, `app`).
2. **Dual-Level Passage + Document Indexing**:
   - Rather than relying solely on whole-document TF-IDF vectors (which dilute short, buried facts in longer articles), the index vectors both the **entire document** and each **individual passage/sentence**.
   - Buried facts (e.g. "15 minutes", "10,000 rows", "60 requests per minute") gain high precision without losing global document context.
3. **Typed Data Models**:
   - Added typed data models (`QueryResult`, `Source`, `EvidenceItem`, `RetrievalScore`) for clean interfaces and predictable data contracts.
4. **Encoding & Safety Robustness**:
   - Uses standard 32-bit Unicode emoji representations to prevent surrogate encoding issues on Windows environments.

---

## Evaluation Benchmark & Threshold Sweep

Evaluation run on the 31 labelled evaluation questions:

| Metric | Result |
|---|---|
| Retrieval accuracy (best-ranked article is the expected one) | 17/17 (100.0%) |
| Exact-fact accuracy (answer contains the expected fact) | 15/15 (100.0%) |
| False-answer rate (answered when it should not have been) | 0/16 (0.0%) |
| Abstention precision | 6/6 (100.0%) |
| Abstention recall | 6/6 (100.0%) |
| Ambiguity handling (PARTIAL or AMBIGUOUS when expected) | 4/4 (100.0%) |
| Answerable questions answered correctly | 15/15 (100.0%) |
| Exact status match | 31/31 (100.0%) |

### Threshold Sweep Results (`python test_app.py --sweep`)

The threshold sweep runs across a grid of $T1$ (minimum retrieval score) and $T2$ (minimum evidence coverage) against both the 31 labelled questions and 10 unseen "trap" questions:

```text
T1 retrieval score (baseline T2 = 0.75)
  0.02  false=0 traps=0/10 answered=15/15 status_match=31/31
  0.05  false=0 traps=0/10 answered=15/15 status_match=31/31
  0.10  false=0 traps=0/10 answered=15/15 status_match=31/31
  0.15  false=0 traps=0/10 answered=15/15 status_match=31/31  <-- Chosen T1
  0.20  false=0 traps=0/10 answered=15/15 status_match=31/31
  0.30  false=0 traps=0/10 answered=15/15 status_match=30/31
  0.40  false=0 traps=0/10 answered=15/15 status_match=29/31
  0.50  false=0 traps=0/10 answered=10/15 status_match=24/31

T2 coverage (baseline T1 = 0.15)
  0.34  false=1 traps=1/10 answered=15/15 status_match=30/31
  0.50  false=1 traps=1/10 answered=15/15 status_match=30/31
  0.60  false=0 traps=1/10 answered=15/15 status_match=31/31
  0.67  false=0 traps=0/10 answered=15/15 status_match=31/31
  0.75  false=0 traps=0/10 answered=15/15 status_match=31/31  <-- Chosen T2
  0.80  false=0 traps=0/10 answered=15/15 status_match=31/31
  0.90  false=0 traps=0/10 answered=14/15 status_match=30/31
  1.00  false=0 traps=0/10 answered=14/15 status_match=30/31
```

- **T1 Choice (0.15)**: Provides a safety floor against irrelevant content while remaining resilient up to 0.20.
- **T2 Choice (0.75)**: Sits in the optimal zero-error window (0.67–0.80) where 100% of answerable questions pass and 0 false answers occur on both the evaluation set and trap questions.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Test Suite
```bash
pytest -q
```

### 4. Run the Evaluation Report & Threshold Sweep
```bash
python test_app.py
python test_app.py --sweep
```
