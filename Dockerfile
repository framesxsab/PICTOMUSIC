FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit's default port
EXPOSE 7860

# HF Spaces expects port 7860
CMD ["streamlit", "run", "src/app1.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
