# Ultra-lightweight Python slim image
FROM python:3.11-slim

# Environment variables to optimize Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install ONLY essential system tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# INSTALL ONLY the CrewAI CLI (very lightweight)
RUN pip install --no-cache-dir crewai

# Set working directory
WORKDIR /app

# Copy all required source files
COPY pyproject.toml ./  
COPY src/ ./src/
COPY knowledge/ ./knowledge/
COPY lib/ ./lib/

# Expose port for Gradio
EXPOSE 7860

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
