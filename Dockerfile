# ---- Build stage ----
FROM cgr.dev/chainguard/python:latest-dev AS builder

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# ---- Runtime stage ----
FROM cgr.dev/chainguard/python:latest

WORKDIR /app

COPY --from=builder /app/deps /app/deps
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

ENV PYTHONPATH=/app/deps:/app/src

ENTRYPOINT ["python"]
CMD ["-c", "\
import subprocess; \
subprocess.run(['python', 'src/data/make_dataset.py'], check=True); \
subprocess.run(['python', 'src/features/build_features.py'], check=True); \
subprocess.run(['python', 'src/models/train_model.py'], check=True) \
"]
