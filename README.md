# A Semantic Evaluation & Analysis Framework for AI Systems

This repository contains a framework utilizing **Sentence Embeddings** and **Cosine Similarity** to quantify the relationship between text inputs. By moving beyond keyword matching, this system can evaluate AI accuracy and perform intelligent log parsing based on **semantic intent**.

## Repository Structure

* `examples/basic_qa.py`: Demonstrates Golden Response testing for Chatbots/LLMs.
* `examples/api_logs.py`: Demonstrates semantic filtering of complex AWS/Cloud logs.
* `requirements.txt`: Project dependencies (pandas, sentence-transformers, scikit-learn).

---

## Core Concept: Semantic Similarity

The framework converts raw text into high-dimensional vectors (embeddings). We then calculate the **Cosine Similarity** to determine how close two pieces of text are in meaning.

### Why this beats Keyword Search:

* **Intent Recognition:** It understands that "expired token" and "authorization error" are semantically related to "Security Failure."
* **Noise Reduction:** It can filter through "noisy" system logs to find specific types of failures without needing complex Regular Expressions (Regex).

---

## Application Scenarios

### 1. Basic QA & Agent Evaluation (`basic_qa.py`)

Used to verify if an AI Agent or Chatbot is providing accurate information.

* **Method:** Compare the **Actual AI Response** against a **"Golden" Reference**.
* **Use Case:** Ensuring the **Obu Eats** assistant correctly explains its focus on "preventive health" rather than just "food delivery."

### 2. Intelligent Log Analysis (`api_logs.py`)

Used to hunt for specific issues within massive, unstructured log files (e.g., AWS Lambda, CloudWatch).

* **Method:** Define a **Semantic Intent** (e.g., "Authentication failures or timeouts") and rank logs by how closely they match that description.
* **Use Case:** Finding "InvalidIdentityToken" or "ExecutionTimedOut" errors without knowing the exact error string beforehand.

### 3. Source Evaluation (RAG Grounding)

Used to detect "Hallucinations" in Retrieval-Augmented Generation systems.

* **Method:** Compare the **LLM Response** against the **Retrieved Source Text**.
* **Low Score:** Indicates the AI is straying from the facts provided in the source documents.

---

## Metric Thresholds for Business

| Similarity Score | Interpretation | Action Required |
| --- | --- | --- |
| **0.90 – 1.00** | Identical / Near-Perfect | None (Pass) |
| **0.75 – 0.89** | Semantically Aligned | Review for minor phrasing or detail gaps |
| **0.50 – 0.74** | Partially Related | High risk of hallucination or partial log match |
| **Below 0.50** | Unrelated / Off-topic | **Failure** (System needs prompt or filter tuning) |

---

## Quick Start

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Log Analysis Example

This script parses raw AWS logs and identifies authentication and timeout issues using semantic intent:

```bash
python examples/api_logs.py
```