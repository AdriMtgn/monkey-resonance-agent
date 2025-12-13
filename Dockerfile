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
