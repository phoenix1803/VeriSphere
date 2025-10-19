# Multi-stage Dockerfile for VeriSphere

# Base stage with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r verisphere && useradd -r -g verisphere verisphere

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/out/logs /app/out/reports && \
    chown -R verisphere:verisphere /app

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov pytest-asyncio pytest-mock black flake8 mypy

# Expose port
EXPOSE 8000

# Switch to non-root user
USER verisphere

# Default command for development
CMD ["python", "-m", "uvicorn", "scripts.run_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production API stage
FROM base as api

# Remove development files
RUN rm -rf tests/ .git/ .github/ docs/ *.md

# Set production environment
ENV ENVIRONMENT=production \
    DEBUG=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Switch to non-root user
USER verisphere

# Production command
CMD ["python", "-m", "uvicorn", "scripts.run_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Worker stage
FROM base as worker

# Remove unnecessary files
RUN rm -rf tests/ .git/ .github/ docs/ *.md

# Set production environment
ENV ENVIRONMENT=production \
    DEBUG=false

# Health check for worker
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Switch to non-root user
USER verisphere

# Worker command
CMD ["python", "-m", "defame.worker.main"]

# Testing stage
FROM base as testing

# Install test dependencies
RUN pip install pytest pytest-cov pytest-asyncio pytest-mock coverage

# Copy test files
COPY tests/ tests/

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=defame", "--cov-report=xml"]

# Documentation stage
FROM node:18-alpine as docs

WORKDIR /docs

# Copy documentation files
COPY docs/ .
COPY README.md .

# Install documentation dependencies
RUN npm install -g @gitbook/cli

# Build documentation
RUN gitbook build

# Nginx stage for serving static files
FROM nginx:alpine as nginx

# Copy built documentation
COPY --from=docs /docs/_book /usr/share/nginx/html/docs

# Copy web UI static files
COPY defame/web/static /usr/share/nginx/html/static

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]