# Use base image with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install uvicorn fastapi apscheduler
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Make sure outputs/model/ and feedback.csv exist
RUN mkdir -p outputs/model
RUN touch feedback.csv

# Expose the API port
EXPOSE 8001

# Start both API server and schedule retraining
CMD ["bash", "-c", "python3 schedule_retrain.py & python3 app.py"]
