# Multi-stage Dockerfile for Home Assistant ML Predictor
# Optimized for production deployment with security hardening

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy requirements first for better layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r haml && useradd -r -g haml -s /bin/bash -m haml

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /venv /venv

# Add venv to PATH
ENV PATH="/venv/bin:$PATH"

# Copy application code
COPY --chown=haml:haml src/ ./src/
COPY --chown=haml:haml config/ ./config/
COPY --chown=haml:haml scripts/ ./scripts/

# Create directories for logs and data with proper permissions
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R haml:haml /app

# Switch to non-root user
USER haml

# Environment variables for production
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH="/app/config" \
    LOG_LEVEL="INFO"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.main_system"]