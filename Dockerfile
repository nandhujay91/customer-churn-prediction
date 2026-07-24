# ---- Stage 1: Builder ----
FROM cgr.dev/chainguard/python:latest-dev AS builder

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# ---- Stage 2: Test ----
FROM builder AS test

ENV PYTHONPATH=/app/deps:/app/src

COPY tests/ ./tests/
COPY data/raw/ ./data/raw/

RUN pip install --no-cache-dir --target=/app/deps pytest ruff
RUN PYTHONPATH=/app/deps python -m pytest tests/ -v
RUN PYTHONPATH=/app/deps python -m ruff check src/

# ---- Stage 3: Runtime ----
FROM cgr.dev/chainguard/python:latest AS runtime

WORKDIR /app

# Depends on the "test" stage completing successfully first —
# if tests fail, this stage is never reached and the build fails.
COPY --from=test /app/deps /app/deps
COPY --from=test /app/src ./src
COPY --from=test /app/configs ./configs

ENV PYTHONPATH=/app/deps:/app/src

ENTRYPOINT ["python"]
CMD ["-c", "\
import subprocess; \
subprocess.run(['python', 'src/data/make_dataset.py'], check=True); \
subprocess.run(['python', 'src/features/build_features.py'], check=True); \
subprocess.run(['python', 'src/models/train_model.py'], check=True) \
"]
