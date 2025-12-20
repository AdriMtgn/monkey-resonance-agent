from typing import List, Union, Dict, Any
from pydantic import BaseModel

import logging

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from langchain_mcp_adapters.client import MultiServerMCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        MCP_URL: str = "http://192.168.1.48:8000"
        OLLAMA_URL: str = "http://test-chat_ollama:11434"
        MODEL: str = "llama2:latest"
        TEMPERATURE: float = 0.1

    async def on_startup(self):
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    def __init__(self):
        self.id = "mkr-agent"
        self.name = "mkr-agent"
        self.valves = self.Valves()

        logger.info(f"Initializing MKR Agent with MCP_URL={self.valves.MCP_URL}")

        # MCP client
        self.mcp_client = MultiServerMCPClient({
            "monkey-resonance-mcp": {
                "url": self.valves.MCP_URL,
                "transport": "streamable_http"
            }
        })
        logger.info("MCP client initialized")

        # LLM
        self.llm = ChatOllama(
            model=self.valves.MODEL,
            base_url=self.valves.OLLAMA_URL,
            temperature=self.valves.TEMPERATURE,
        )
        logger.info(f"LLM initialized: {self.valves.MODEL}")

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
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
        self.base_chain = self.prompt | self.llm
        logger.info("Audio engineering agent with MCP tools ready")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        """Main pipeline method - streams audio engineering responses WITH MCP tools"""
        logger.info("Pipe method called")
        
        messages = body.get("messages", [])
        if not messages:
            yield {"type": "text", "text": "No messages provided"}
            return

        logger.info(f"Processing {len(messages)} messages")
        chat_history = self._convert_messages(messages)
        user_input = messages[-1]["content"]
        logger.info(f"User input: {user_input[:100]}...")

        # ✅ FIXED: Get tools ASYNC
        try:
            tools = self._get_tools()
            logger.info(f"Loaded {len(tools)} MCP tools")
            
            if tools:
                chain = self.base_chain.bind_tools(tools)
            else:
                chain = self.base_chain
                logger.warning("No MCP tools available")
        except Exception as e:
            logger.error(f"Tool loading failed: {e}")
            chain = self.base_chain
            tools = []

        try:
            # Stream response
            for chunk in chain.stream({
                "input": user_input,
                "chat_history": chat_history
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield {"type": "text", "text": chunk.content}
                    
                    # ✅ FIXED: Safe effects detection
                    if "effects" in chunk.content.lower() and len(chunk.content.split()) > 1:
                        try:
                            words = chunk.content.split()
                            input_index = int(words[1][0]) if words[1][0].isdigit() else 1
                            logger.info(f"Fetching effects for input {input_index}")
                            
                            # ✅ Effects via MCP (non-async for simplicity)
                            effects_data = self.mcp_client.access_resource(
                                server_name="monkey-resonance-mcp",
                                uri=f"/list_effects/{input_index}"
                            )
                            yield {"type": "text", "text": f"\n🎛️ Effects for input {input_index}: {effects_data}"}
                        except Exception as e:
                            logger.error(f"Effects error: {e}")

            logger.info("Pipe completed successfully")
        except Exception as e:
            logger.error(f"Pipe error: {str(e)}", exc_info=True)
            yield {"type": "text", "text": f"Audio system error: {str(e)}"}

    @staticmethod
    def pipes():
        return [{"id": "mkr-agent", "name": "MKR Audio Agent 🎛️"}]

    async def _get_tools(self) -> List:
        """✅ FIXED: Async MCP tools"""
        logger.info("Fetching MCP tools")
        try:
            # ✅ AWAIT the coroutine
            mcp_tools = await self.mcp_client.get_tools(
                server_name="monkey-resonance-mcp"
            )
            logger.info(f"Retrieved {len(mcp_tools)} MCP tools")

            langchain_tools = []
            for mcp_tool in mcp_tools:
                logger.info(f"Creating tool: {mcp_tool.name}")
                
                @tool(mcp_tool.name, description=mcp_tool.description)
                async def mcp_tool_wrapper(**kwargs):  # ✅ Async wrapper
                    logger.info(f"Calling MCP tool {mcp_tool.name}: {kwargs}")
                    try:
                        # ✅ AWAIT tool call
                        result = await self.mcp_client.call_tool(
                            server_name="monkey-resonance-mcp",
                            name=mcp_tool.name,
                            arguments=kwargs
                        )
                        return result
                    except Exception as e:
                        logger.error(f"MCP tool error: {e}")
                        raise

                langchain_tools.append(mcp_tool_wrapper)
            
            return langchain_tools
        except Exception as e:
            logger.error(f"MCP tools failed: {e}")
            return []

    def _convert_messages(self, messages: List[dict]) -> List:
        history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] in ("assistant", "ai"):
                history.append(AIMessage(content=msg["content"]))
        return history

    def get_model(self) -> Dict[str, Any]:
        return {
            "model": self.valves.MODEL,
            "label": self.name,
            "type": "pipe",
            "description": "🎛️ MCP-powered audio equipment manager with real-time sound card control"
        }

    async def get_tools(self) -> List[Dict[str, Any]]:
        """✅ ASYNC tools list for OpenWebUI"""
        tools = await self._get_tools()
        return [
            {
                "name": getattr(tool, 'name', 'unknown'),
                "description": getattr(tool, 'description', 'MCP tool'),
                "parameters": getattr(tool, 'args', {})
            }
            for tool in tools
        ]
