FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Add near the top, after the FROM line
RUN apt-get update && apt-get install -y coreutils && rm -rf /var/lib/apt/lists/*

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for the stock prediction system
RUN mkdir -p /app/src \
             /app/data/price_data \
             /app/models \
             /app/output \
             /app/logs \
             /app/processed_data \
             /app/cache \
             /app/notebooks

# Copy the stock prediction system files
COPY src/ /app/src/

# Create default config files if they don't exist
RUN touch /app/config.json /app/.env

# Copy configuration files if they exist (will overwrite the empty files)
COPY config.json /app/config.json
COPY .env /app/.env

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
RUN chmod -R 755 /app
USER appuser

# Default command (can be overridden)
CMD ["python", "/app/src/main.py", "--help"]