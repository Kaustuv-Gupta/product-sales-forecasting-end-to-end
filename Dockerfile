# -------------------------------
# Base Image
# -------------------------------
FROM python:3.11-slim

# -------------------------------
# Set working directory
# -------------------------------
WORKDIR /product-sales-forecast

# -------------------------------
# Install system dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Copy requirements first
# -------------------------------
COPY requirements/base.txt requirements/base.txt
COPY requirements/ml.txt requirements/ml.txt
COPY requirements/api.txt requirements/api.txt

# -------------------------------
# Install requirements
# -------------------------------
RUN pip install --no-cache-dir -r requirements/base.txt \
    && pip install --no-cache-dir -r requirements/ml.txt \
    && pip install --no-cache-dir -r requirements/api.txt

# -------------------------------
# Copy application code
# -------------------------------
COPY app/ app/
COPY artifacts/ artifacts/
COPY ml/ ml/

# -------------------------------
# Copy environment file
# -------------------------------
COPY .env .env

# -------------------------------
# Expose port
# -------------------------------
EXPOSE 8000

# -------------------------------
# Run FastAPI (dev mode as requested)
# -------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--env-file", ".env"]