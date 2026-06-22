FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY src/ src/
COPY tests/ tests/

# Expose port for FastAPI or MLflow
EXPOSE 8000
EXPOSE 5000

# Default command is to run the FastAPI API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
