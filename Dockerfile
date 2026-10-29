# Stage 1: Build virtual environment and train model
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create virtual environment and install production dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and training data
COPY supportpilot/ supportpilot/
COPY data/ data/

# Train triage model (generates models/triage_model.joblib)
RUN python -m supportpilot.triage --train

# Optimize virtual environment to keep final image well under 400 MB:
# Strip binary symbols, remove caches, and uninstall build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends binutils && \
    find /opt/venv -type f -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find /opt/venv -name "*.pyc" -delete && \
    /opt/venv/bin/pip uninstall -y pip setuptools && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Minimal runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create dedicated non-root user (no curl installed to minimize attack surface)
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code, migrations, and model with non-root ownership
COPY --chown=appuser:appuser supportpilot/ supportpilot/
COPY --chown=appuser:appuser migrations/ migrations/
COPY --chown=appuser:appuser alembic.ini .
COPY --from=builder --chown=appuser:appuser /app/models/ models/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Healthcheck using Python standard library (no curl needed in runtime image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

CMD ["sh", "-c", "alembic upgrade head && echo \"migrations complete\" && exec uvicorn supportpilot.api.main:app --host 0.0.0.0 --port 8000 --no-access-log --log-level $(echo ${LOG_LEVEL:-info} | tr '[:upper:]' '[:lower:]' | sed 's/^warn$/warning/')"]
