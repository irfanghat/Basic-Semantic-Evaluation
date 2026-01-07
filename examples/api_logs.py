import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("api-log-lookup")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.propagate = False


# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Simulated insane AWS logs
# -----------------------------
raw_logs = """
2025-01-06T14:03:21.112Z INFO  Lambda.Start RequestId=9d2a runtime=nodejs18.x
2025-01-06T14:03:21.118Z DEBUG API-Gateway Received request GET /v1/meals
2025-01-06T14:03:21.120Z INFO  CognitoJWTVerifier Token parsed successfully
2025-01-06T14:03:21.121Z WARN  DynamoDB ThrottlingException: Rate exceeded
2025-01-06T14:03:21.125Z INFO  Retrying DynamoDB call (attempt 2)
2025-01-06T14:03:22.301Z ERROR Lambda Task timed out after 6.00 seconds
2025-01-06T14:03:22.302Z INFO  Lambda.End RequestId=9d2a

2025-01-06T14:04:10.998Z INFO  StepFunctions.ExecutionStarted arn=arn:aws:states:...
2025-01-06T14:04:11.004Z ERROR States.TaskFailed Cause=InvalidIdentityToken
2025-01-06T14:04:11.005Z WARN  Cognito Authorizer failed: token expired
2025-01-06T14:04:11.006Z DEBUG Falling back to anonymous user
2025-01-06T14:04:11.006Z ERROR API-Gateway Execution failed due to authorization error

2025-01-06T14:05:44.771Z INFO  Lambda.Start RequestId=ab81
2025-01-06T14:05:44.772Z ERROR UnhandledPromiseRejection: TypeError: Cannot read property 'userId'
2025-01-06T14:05:44.773Z WARN  Possible malformed request body
2025-01-06T14:05:44.774Z INFO  Lambda.End RequestId=ab81

2025-01-06T14:06:59.432Z ERROR StepFunctions.ExecutionTimedOut
2025-01-06T14:06:59.433Z INFO  Compensation workflow triggered
2025-01-06T14:06:59.434Z WARN  SQS backlog size exceeded threshold
"""

# -----------------------------
# Split logs into rows
# -----------------------------
log_lines = [
    line.strip()
    for line in raw_logs.split("\n")
    if line.strip()
]

df = pd.DataFrame(log_lines, columns=["log"])

# -----------------------------
# Define semantic intent
# -----------------------------
target_intent = """
Authentication or authorization failures, invalid or expired tokens,
identity verification errors, and execution timeouts in AWS Lambda
or Step Functions workflows.
"""

# -----------------------------
# Create embeddings
# -----------------------------
log_embeddings = model.encode(df["log"].tolist())
target_embedding = model.encode([target_intent])

# -----------------------------
# Similarity scoring
# -----------------------------
df["similarity"] = cosine_similarity(
    log_embeddings,
    target_embedding
).ravel()

# -----------------------------
# Rank most relevant logs
# -----------------------------
df_sorted = df.sort_values("similarity", ascending=False)

pd.set_option("display.max_colwidth", None)

logger.info("\nMost Relevant Authentication / Timeout Issues \n")
print(df_sorted.head(8).to_string(index=False))
