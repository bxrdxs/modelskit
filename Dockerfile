# =============================================================================
# AINA Translator – GPU-ready Docker image
# =============================================================================
# Base: NVIDIA CUDA 12.1 runtime on Ubuntu 22.04
# Falls back to CPU automatically if no GPU is present.
# =============================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.1 support first (big layer, cached separately)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# HuggingFace cache inside the container
ENV HF_HOME=/app/.hf_cache
ENV TRANSFORMERS_CACHE=/app/.hf_cache

# Expose the HTTP port
EXPOSE 8000

# Health-check (optional, useful for orchestrators)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
