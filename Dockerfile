# ---- Build stage ----
FROM cgr.dev/chainguard/python:latest-dev AS builder

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

# ---- Runtime stage ----
FROM cgr.dev/chainguard/python:latest

WORKDIR /app

COPY --from=builder /app/deps /app/deps
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

# data/models are mounted at runtime, not baked into the image
ENV PYTHONPATH=/app/deps:/app/src

# Default: run the full pipeline (clean -> features -> train)
# Override CMD per step if you only want one stage
ENTRYPOINT ["python"]
CMD ["-c", "\
import subprocess; \
subprocess.run(['python', 'src/data/make_dataset.py', 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv', 'data/processed/churn_cleaned.csv'], check=True); \
subprocess.run(['python', 'src/features/build_features.py', 'data/processed/churn_cleaned.csv', 'data/processed'], check=True); \
subprocess.run(['python', 'src/models/train_model.py', 'data/processed', 'models'], check=True) \
"]
