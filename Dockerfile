FROM ghcr.io/open-webui/pipelines:main

# Install REQUIRED dependencies for MCP Agent (not in base image)
RUN pip install --no-cache-dir \
    langchain \
    langchain-ollama \
    langchain-core \
    langchain-community \
    langchain-mcp-adapters \
    pydantic \
    typing-extensions

# Copy your pipeline script
COPY agent.py /app/pipelines/agent.py

# Expose standard Pipelines port
EXPOSE 9099

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    curl -f http://localhost:9099/ || exit 1
