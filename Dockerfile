FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create reports directory
RUN mkdir -p reports

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create an entrypoint script
RUN echo '#!/bin/sh\n\
if [ "$1" = "weekly" ]; then\n\
    python automated_reports.py weekly\n\
elif [ "$1" = "monthly" ]; then\n\
    python automated_reports.py monthly\n\
else\n\
    echo "Invalid argument. Use: weekly or monthly"\n\
    exit 1\n\
fi' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 