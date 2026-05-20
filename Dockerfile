# Use an official, lightweight Python base image
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr (helps with logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies.
# - libgomp1 is a requirement for OpenMP, which is used by XGBoost and LightGBM.
# - apt-cache is cleaned up in the same layer to minimize image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to optimize Docker build caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files to the container
COPY . .

# Expose the port Streamlit runs on (default: 8501)
EXPOSE 8501

# Health check to monitor the health of the Streamlit application
# We use Python's built-in urllib to avoid installing curl/wget, keeping the image small.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Command to run the Streamlit application when the container starts.
# Runs on port 8501 and binds to 0.0.0.0 to receive external requests.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
