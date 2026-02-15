FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env.example ./.env.example

# data/ is mounted as volume, not copied
# logs/ is created at runtime

CMD ["python", "-m", "app"]
