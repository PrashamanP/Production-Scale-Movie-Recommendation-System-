# syntax=docker/dockerfile:1.4

# ============================================
# Stage 1: Base image with all dependencies
# ============================================
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies for implicit (ALS library) and kcat
# Use BuildKit cache mount for faster apt operations
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    kcat \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY monitoring/ ./monitoring/

# Copy training scripts
COPY scripts/ ./scripts/

# ============================================
# Stage 2: Test image (includes test tools)
# ============================================
FROM base AS test

# Install test dependencies with BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-cov==4.1.0

# Copy test files
COPY tests/ ./tests/
COPY .coveragerc ./
COPY pyproject.toml ./

# Run tests (will be overridden in Jenkinsfile, but good for manual builds)
CMD ["pytest", "-v", "--cov=src", "--cov-report=term-missing", "tests"]

# ============================================
# Stage 3: Production image (minimal)
# ============================================
FROM base AS production

# Create empty artifacts directory (models will be mounted from host at runtime)
RUN mkdir -p ./artifacts

# Set build metadata (will be injected by Jenkins)
ARG BUILD_NUMBER=dev
ARG GIT_COMMIT=unknown
ARG BUILD_TIMESTAMP=unknown
ENV BUILD_NUMBER=${BUILD_NUMBER}
ENV GIT_COMMIT=${GIT_COMMIT}

# Add OCI-compliant labels for better provenance
LABEL org.opencontainers.image.created="${BUILD_TIMESTAMP}" \
      org.opencontainers.image.authors="Team Blockbusters - Team 13" \
      org.opencontainers.image.url="https://github.com/cmu-seai/group-project-f25-blockbusters" \
      org.opencontainers.image.source="https://github.com/cmu-seai/group-project-f25-blockbusters" \
      org.opencontainers.image.version="${BUILD_NUMBER}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.title="Movie Recommender Service" \
      org.opencontainers.image.description="ALS-based movie recommendation service for 17-645 ML in Production"

# Expose port
EXPOSE 8082

# Start the application with Gunicorn (production server)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8082", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "src.serve_als_model:create_app()"]
