FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MQTT client
RUN pip install paho-mqtt

# Copy HDC source code
COPY hdc/ ./hdc/
COPY federated_train.py .
COPY federated_aggregator.py .

# Create data directories
RUN mkdir -p /app/data /app/test_data

# Set Python path
ENV PYTHONPATH=/app

# Default command (will be overridden)
CMD ["python", "federated_train.py"]