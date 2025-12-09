# 1. Save as MCPAgentPipe.py
# 2. Create requirements.txt
echo "langchain-ollama
langchain-mcp-adapters
langchain
pydantic" > requirements.txt

# 3. Docker Swarm (your existing network)
docker run -d \
  --name mcp-pipeline \
  -p 9099:9099 \
  -v $(pwd)/MCPAgentPipe.py:/app/pipelines/MCPAgentPipe.py \
  -v $(pwd)/requirements.txt:/app/requirements.txt \
  --network your-swarm-net \
  --restart always \
  ghcr.io/open-webui/pipelines:main

# 4. OpenWebUI → Settings → OpenAI API → http://mcp-pipeline:9099
