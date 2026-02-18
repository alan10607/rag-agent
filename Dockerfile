FROM python:3.14-slim

WORKDIR /app

# Install system dependencies (curl for Cursor CLI installer)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Cursor Agent CLI
RUN curl https://cursor.com/install -fsS | bash && \
    ln -sf /root/.local/bin/agent /usr/local/bin/agent

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ragent/ ./ragent/

# Copy MCP configuration for Cursor Agent
COPY ragent/mcp/mcp.json /root/.cursor/mcp.json

# data/ is mounted as volume, not copied
# logs/ is created at runtime

# Copy entrypoint script and make it executable
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

CMD ["python", "-m", "ragent"]
