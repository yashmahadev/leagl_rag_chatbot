# -------------------------------
# Stage 1: Base Python Environment
# -------------------------------
    FROM python:3.10-slim

    # Prevent interactive prompts during install
    ENV DEBIAN_FRONTEND=noninteractive
    ENV PYTHONUNBUFFERED=1
    
    # Set working directory
    WORKDIR /app
    
    # Install system dependencies (minimal)
    RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        wget \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy dependency list
    COPY requirements.txt .
    
    # Install Python dependencies efficiently
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy the full project
    COPY . .
    
    # Expose Streamlit port
    EXPOSE 8501
    
    # Streamlit environment setup
    ENV STREAMLIT_SERVER_HEADLESS=true
    ENV STREAMLIT_SERVER_PORT=8501
    ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
    ENV OLLAMA_HOST=http://host.docker.internal:11434
    
    # Disable telemetry/logging for smaller container logs
    ENV STREAMLIT_TELEMETRY=False
    
    # Run Streamlit app
    CMD ["streamlit", "run", "app_chatbot.py"]
    