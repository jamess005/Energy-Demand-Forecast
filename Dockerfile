FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Copy training code (needed for model class imports) and model artefacts
COPY training/train.py ./training/train.py
COPY training/models/lstm_bias_table.json ./training/models/lstm_bias_table.json
COPY training/models/lstm_feature_scaler.json ./training/models/lstm_feature_scaler.json
COPY training/models/lstm_norm_stats.json ./training/models/lstm_norm_stats.json

# Copy LSTM checkpoint (mount at runtime or copy here for deployment)
# COPY training/models/lstm-v2-epoch=08-val_MAE=1364.ckpt ./training/models/

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API via uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]