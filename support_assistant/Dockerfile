# Zepto Data & AI Platform - Module 3: Support Assistant Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy support assistant application files
COPY . .

# Set environment variables (MOCK_LLM=1 default graded baseline)
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1
ENV PORT=7860

# Expose service port
EXPOSE 7860

# Serve FastAPI app via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
