import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

input = """
Obu Eats is an AI-powered nutrition and meal planning platform based in Kenya. 
Unlike a standard food delivery app, like Uber Eats, it focuses on preventive health.
"""

df = pd.DataFrame(
    [(input,)],
    columns=["response"]
)

expected_response = """
In summary, Obu Eats is an AI-powered health and nutrition platform focused on the Kenyan market. 
It is designed to bridge the gap between nutrition data and daily eating habits.
"""

embeddings = model.encode(df["response"].tolist())
target_embedding = model.encode([expected_response])

df["similarity"] = cosine_similarity(embeddings, target_embedding).ravel()
print(df)
