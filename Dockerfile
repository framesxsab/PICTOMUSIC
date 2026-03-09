FROM python:3.11-slim
# Build: 2026-03-10 v3 -- set XSRF/CORS via ENV

# Create a new user with UID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR $HOME/app

# Install system dependencies as root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first, setting correct ownership
COPY --chown=user:user requirements.txt .

# Switch to the non-root user
USER user

# Install python dependencies to the user's local directory
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy the rest of the application files with user ownership
COPY --chown=user:user . .

# Expose Streamlit's default port
EXPOSE 7860

# HF Spaces expects port 7860
CMD ["streamlit", "run", "src/app1.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
