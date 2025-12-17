from pydantic import BaseModel
from typing import Any, Dict, Generator
import asyncio

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_mcp_adapters.client import MultiServerMCPClient


class Valves(BaseModel):
    MCP_URL: str = "http://192.168.1.48:8000"
    OLLAMA_URL: str = "http://test-chat_ollama:11434"
    MODEL: str = "llama2:latest"
    TEMPERATURE: float = 0.1


class Pipeline:
    def __init__(self):
        self.name = "mkr-agent"
        self.type = "pipe"
        self.valves = Valves()

        # MCP client
        self.mcp_client = MultiServerMCPClient({
            "monkey-resonance-mcp": {
                "url": self.valves.MCP_URL,
                "transport": "streamable_http"
            }
        })

        # LLM
        llm = ChatOllama(
            model=self.valves.MODEL,
            base_url=self.valves.OLLAMA_URL,
            temperature=self.valves.TEMPERATURE,
        )

        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a specialized audio equipment manager agent. "
                "You control sound cards, apply effects, manage volumes, and handle recordings. "
                "Use the available tools to perform these tasks. "
                "Respond in a musical and technical manner, using audio engineering terminology."
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])

        # 🔑 LCEL tool binding - we'll bind tools in the pipe method
        self.chain = (
            prompt
            | llm
        )

        # Store tools for later use
        self.tools = []
    
    def _get_tools(self):
        """Get tools from MCP client"""
        try:
            # Get tools from MCP client
            tools = self.mcp_client.get_tools(
                server_name="monkey-resonance-mcp"
            )
            return tools
        except Exception as e:
            print(f"Error getting tools: {e}")
            return []
    
    def _convert_messages(self, messages):
        history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] in ("assistant", "ai"):
                history.append(AIMessage(content=msg["content"]))
        return history

    async def pipe(self, body: Dict[str, Any]) -> Generator[str, None, None]:
        messages = body.get("messages", [])
        if not messages:
            yield "No messages provided"
            return

        chat_history = self._convert_messages(messages)
        user_input = messages[-1]["content"]

        # Get tools for this invocation
        tools = self._get_tools()
        
        try:
            result = self.chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Handle effects queries
            if "effects" in result.content.lower():
                try:
                    # Extract input index from response (assuming format like "Input 2:")
                    input_index = int(result.content.split()[1][0])
                    effects_data = self.mcp_client.access_resource(
                        server_name="monkey-resonance-mcp",
                        uri=f"/list_effects/{input_index}"
                    )
                    yield f"\nEffect details for input {input_index}: {effects_data}"
                except (IndexError, ValueError, Exception) as e:
                    yield f"\nError retrieving effects: {str(e)}"

            yield result.content
        except Exception as e:
            # Handle any errors gracefully
            yield f"Error in pipe execution: {str(e)}"
            return

    def get_tools(self):
        """Return the tools available to this pipeline"""
        # This is a placeholder - the actual tools should be loaded in __init__
        return []
