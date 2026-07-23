FROM cgr.dev/chainguard/python:latest-dev AS builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt
COPY src/ ./src/

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=builder /app/deps /app/deps
COPY --from=builder /app/src /app/src
ENV PYTHONPATH=/app/deps:/app/src
ENTRYPOINT ["python", "-m", "src.models.train_model"]
