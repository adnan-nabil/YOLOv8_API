# 1. Use a slim, specific base image for minimal footprint and security vulnerabilities
FROM python:3.10-slim

# 2. Set system environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# 3. Install minimal system dependencies required by OpenCV / Ultralytics
# Clean up apt caches in the same layer to save space
# Install modern system dependencies required by OpenCV / Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set a dedicated working directory
WORKDIR /workspace

# 5. Leverage Docker layer caching for heavy ML dependencies
# This layer ONLY rebuilds if requirements.txt changes, saving massive build times
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code and weights AFTER dependencies are baked in
COPY ./app ./app

# 7. Create a non-root system user for runtime security compliance
RUN useradd -u 8888 appuser && chown -R appuser:appuser /workspace
USER appuser

# 8. Document the network interface port mapping
EXPOSE 7860

# 9. Run with Uvicorn, explicitly binding to all interfaces on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]