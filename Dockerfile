FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0t64 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt demo_requirements.txt ./
RUN pip install --no-cache-dir -r demo_requirements.txt

# Copy application code
COPY app.py predict.py moire_feature.py ./
COPY model/ model/
COPY templates/ templates/
COPY static/ static/

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
