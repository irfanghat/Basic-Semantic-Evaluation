# A Basic Semantic Evaluation Framework for AI Systems

This framework utilizes **Sentence Embeddings** and **Cosine Similarity** to quantify how closely an AI's output matches a "Golden" or "Reference" response. Unlike traditional keyword matching, this approach understands the *meaning* (semantics) behind the words.

---

## Core Concept: Semantic Similarity Evaluation

The provided script converts text into high-dimensional vectors (embeddings) where the distance between vectors represents the difference in meaning.

### Why Use Cosine Similarity?

* **Context over Keywords:** It recognizes that "preventive health" and "wellness and longevity" are related, even though the words are different.
* **Scale Invariance:** It measures the **angle** between vectors rather than length, meaning a short summary can be highly similar to a longer, detailed explanation if they share the same intent.

---

## Application Scenarios

### 1. Agent Evaluation (Task Completion)

AI Agents often have multiple "execution paths." You can use this framework to verify if an agent arrived at the correct conclusion, even if its phrasing varies per run.

* **How to use:** Store the "Success Criteria" as the `expected_response`. If the agent's final answer scores , the task is marked as "Success."

### 2. Chatbot UI Evaluation (Regression Testing)

When updating a UI or changing an LLM model, you need to ensure the "vibes" or "persona" haven't changed.

* **How to use:** * **Baseline:** Capture responses from the stable version.
* **Test:** Capture responses from the new version.
* **Evaluation:** A low similarity score flags a "regression," alerting developers that the UI is presenting information differently than before.



### 3. Source Evaluation (Hallucination Detection)

In RAG (Retrieval-Augmented Generation) systems like Obu Eats, the AI must stay grounded in its provided sources (e.g., medical nutrition papers).

* **How to use:** Compare the **LLM Response** against the **Retrieved Source Text**.
* High similarity suggests the AI is accurately summarizing the source.
* Low similarity might indicate the AI is "hallucinating" or pulling from its internal training data instead of the verified source.



---

## Implementation Guide

The following code demonstrates a "Similarity Test" for the Obu Eats concept:

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------
# Load the Embedding Model
# ------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------
# Define the Actual output
# ------------------------------------
actual_input = """
Obu Eats is an AI-powered nutrition and meal planning platform based in Kenya. 
Unlike a standard food delivery app, like Uber Eats, it focuses on preventive health.
"""

# ----------------------------------------------------
# Define the 'Golden' output (the perfect reference)
# ----------------------------------------------------
expected_response = """
In summary, Obu Eats is an AI-powered health and nutrition platform focused on the Kenyan market. 
It is designed to bridge the gap between nutrition data and daily eating habits.
"""

# ------------------------------------
# Convert to Vectors
# ------------------------------------
embeddings = model.encode([actual_input])
target_embedding = model.encode([expected_response])

# ------------------------------------
# Calculate Score (0.0 to 1.0)
# ------------------------------------
similarity = cosine_similarity(embeddings, target_embedding).ravel()[0]

print(f"Semantic Similarity Score: {similarity:.4f}")

# ------------------------------------
# Output: ~0.82 (Highly Similar)
# ------------------------------------

```

---

## Metric Thresholds for Business

| Similarity Score | Interpretation | Action Required |
| --- | --- | --- |
| **0.90 – 1.00** | Identical or Near-Perfect | None (Pass) |
| **0.75 – 0.89** | Semantically Aligned | Review for minor phrasing issues |
| **0.50 – 0.74** | Partially Related | High risk of hallucination or missed context |
| **Below 0.50** | Unrelated / Off-topic | **Critical Failure** (System needs prompt tuning) |