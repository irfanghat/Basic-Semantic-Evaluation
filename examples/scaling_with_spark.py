import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql.functions import pandas_udf
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Broadcast the model to all workers so it's only loaded once per executor
# ----------------------------------------------------------------------------
model_name = "all-MiniLM-L6-v2"

@pandas_udf("float")
def get_semantic_score_udf(batch_responses: pd.Series) -> pd.Series:
    # ----------------------------------------------------------------------------
    # Load model once inside the worker
    # ----------------------------------------------------------------------------
    model = SentenceTransformer(model_name)
    target_intent = "Authentication or authorization failures..."
    target_emb = model.encode([target_intent])
    
    # ----------------------------------------------------------------------------
    # Process the entire batch at once (Vectorized)
    # ----------------------------------------------------------------------------
    embeddings = model.encode(batch_responses.tolist())
    scores = cosine_similarity(embeddings, target_emb).ravel()
    return pd.Series(scores)

# ----------------------------------------------------------------------------
# Run it on millions of rows in a Delta Table
# ----------------------------------------------------------------------------
df = spark.table("bronze_logs")
enriched_df = df.withColumn("similarity_score", get_semantic_score_udf(df.log_text))
enriched_df.write.saveAsTable("silver_evaluated_logs")