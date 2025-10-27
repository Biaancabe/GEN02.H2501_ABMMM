
# Simple KPI analysis for Study Buddy
# - Reads Excel log file
# - Filters out "__app_start__" system events
# - Creates one clean line chart (Response vs Retrieval latency)

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to Excel file
INPUT_XLSX = Path("data/metrics/technical_metrics_all.xlsx")

# Output directory for charts and summary tables
OUT_DIR = Path("data/metrics_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target times (desired upper limits)
TARGET_RESPONSE_S  = 3.0   # seconds
TARGET_RETRIEVAL_S = 1.0   # seconds

# Read the Excel file and normalize column names
df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Check required columns exist
need = {"response_latency_s", "retrieval_latency_s", "query"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

# ---------- FILTER THE DATA ----------
# Remove rows where the query field contains "__app_start__"
df = df[~df["query"].astype(str).str.lower().str.contains("__app_start__", na=False)]

# Remove empty query rows (just in case)
df = df[df["query"].astype(str).str.strip() != ""].copy()

# 3) Sort chronologically if timestamps are available
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp", na_position="last")

# Create a simple index for plotting on the x-axis
df = df.reset_index(drop=True)
df["q_idx"] = np.arange(1, len(df) + 1)

# Extract the two main latency columns
resp = df["response_latency_s"].astype(float)
retr = df["retrieval_latency_s"].astype(float)

# ---------- CREATE SUMMARY TABLE ----------
summary = pd.DataFrame({
    "KPI": ["Response latency (s)", "Retrieval latency (s)"],
    "Mean (s)":   [resp.mean(),         retr.mean()],
    "Median (s)": [resp.median(),       retr.median()],
    "P95 (s)":    [resp.quantile(0.95), retr.quantile(0.95)],
    "Target Time": [f"≤ {TARGET_RESPONSE_S}s", f"≤ {TARGET_RETRIEVAL_S}s"]
}).round(3)

# Save summary as CSV
summary.to_csv(OUT_DIR / "latency_summary_targettime.csv", index=False)

# ---------- PLOT: LATENCY PER QUERY ----------
plt.figure()
plt.plot(df["q_idx"], resp, label="Response latency (s)")
plt.plot(df["q_idx"], retr, label="Retrieval latency (s)")

# Add dashed lines for target times
plt.axhline(TARGET_RESPONSE_S, linestyle="--", linewidth=1, label=f"Target Time Response {TARGET_RESPONSE_S}s")
plt.axhline(TARGET_RETRIEVAL_S, linestyle="--", linewidth=1, label=f"Target Time Retrieval {TARGET_RETRIEVAL_S}s")

plt.xlabel("Query #")
plt.ylabel("Seconds")
plt.title("Latency per Query")
plt.legend()
plt.tight_layout()

# Save the line chart
plt.savefig(OUT_DIR / "latency_per_query.png", dpi=200)
plt.close()
