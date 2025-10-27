
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def compute_inter_chunk_similarity(embs: np.ndarray) -> float:
    """Compute average cosine similarity between embeddings (redundancy metric)."""
    if embs.shape[0] < 2:
        return 0.0

    # Compute the cosine similarity matrix (dot product, assuming normalized embeddings)
    sims = np.dot(embs, embs.T)
    n = sims.shape[0]

    # Exclude self-similarity terms (the diagonal) when averaging
    return float((np.sum(sims) - n) / (n*n - n))

def log_metrics(query: str, response_latency: float, retrieval_latency: float, inter_chunk_sim: float):
    """Append a row with latency + similarity metrics to a CSV log."""

    # Ensure the directory for metrics exists
    metrics_dir = Path("data/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "technical_metrics.csv"

    # Prepare one log row with rounded numeric values
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "response_latency_s": round(response_latency, 4),
        "retrieval_latency_s": round(retrieval_latency, 4),
        "avg_inter_chunk_similarity": round(inter_chunk_sim, 6)
    }

    # Append to existing file if present, otherwise create a new one with headers
    df = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
