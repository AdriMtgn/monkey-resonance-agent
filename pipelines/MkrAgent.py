from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from typing import Any, Dict, Generator, List
import asyncio
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
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is shutdown.
        print(f"on_shutdown:{__name__}")
        pass

    
    def __init__(self):
        self.id = "mkr-agent"
        self.name = "mkr-agent"
        self.valves = self.Valves()

        logger.info(f"Initializing Pipeline with MCP_URL={self.valves.MCP_URL}, OLLAMA_URL={self.valves.OLLAMA_URL}")

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
        logger.info(f"LLM initialized with model={self.valves.MODEL}, temperature={self.valves.TEMPERATURE}")

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
        logger.info("Prompt template created")

        # Initialize empty tools list - will be populated in pipe method
        self.tools = []
    
        # Store the base chain without tools (tools will be added dynamically)
        self.base_chain = self.prompt | self.llm
        logger.info("Base chain assembled: prompt -> LLM")

def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
    logger.info("Pipe method called")
    logger.debug(f"Input body: {body}")

    messages = body.get("messages", [])
    if not messages:
        logger.warning("No messages provided in body")
        yield {"type": "text", "text": "No messages provided"}
        return

    logger.info(f"Processing {len(messages)} messages")
    chat_history = self._convert_messages(messages)
    user_input = messages[-1]["content"]
    logger.info(f"User input: {user_input}")

    # Get tools for this invocation and update self.tools
    self.tools = self._get_tools()
    logger.info(f"Loaded {len(self.tools)} tools for this invocation")

    try:
        # Create the full chain with tools
        if self.tools:
            logger.info("Binding tools to LLM")
            chain = self.base_chain.bind_tools(self.tools)
            logger.info(f"Chain now includes {len(self.tools)} tools")
        else:
            logger.warning("No tools available - using base chain without tools")
            chain = self.base_chain

        logger.info("Invoking LLM chain")
        logger.debug(f"Chain input: {{'input': '{user_input}', 'chat_history': {chat_history}}}")

        # Stream the response from the LLM
        for chunk in chain.stream({
            "input": user_input,
            "chat_history": chat_history
        }):
            logger.debug(f"Stream chunk: {chunk}")
            if chunk.content:
                yield {"type": "text", "text": chunk.content}

            # Handle effects queries in the stream
            if "effects" in chunk.content.lower():
                logger.info("Detected effects query in response")
                try:
                    # Extract input index from response (assuming format like "Input 2:")
                    input_index = int(chunk.content.split()[1][0])
                    logger.info(f"Extracting effects for input {input_index}")

                    logger.info(f"Calling MCP resource: /list_effects/{input_index}")
                    effects_data = self.mcp_client.access_resource(
                        server_name="monkey-resonance-mcp",
                        uri=f"/list_effects/{input_index}"
                    )
                    logger.info(f"Received effects data: {effects_data}")
                    yield {"type": "text", "text": f"\nEffect details for input {input_index}: {effects_data}"}
                except (IndexError, ValueError, Exception) as e:
                    logger.error(f"Error retrieving effects: {str(e)}", exc_info=True)
                    yield {"type": "text", "text": f"\nError retrieving effects: {str(e)}"}

        logger.info("Pipe execution completed successfully")
    except Exception as e:
        logger.error(f"Error in pipe execution: {str(e)}", exc_info=True)
        yield {"type": "text", "text": f"Error in pipe execution: {str(e)}"}
    
    def pipes(self):
        return [{"id": "mkr-agent", "name": "MKR Agent"}]
    
    def _get_tools(self):
        """Get tools from MCP client and convert to LangChain tools"""
        logger.info("Fetching tools from MCP client")
        try:
            mcp_tools = self.mcp_client.get_tools(
                server_name="monkey-resonance-mcp"
            )
            logger.info(f"Successfully retrieved {len(mcp_tools)} tools from MCP")

            # Convert MCP tools to LangChain tools
            langchain_tools = []
            for mcp_tool in mcp_tools:
                logger.info(f"Converting MCP tool: {mcp_tool.name}")

                # Create a LangChain tool wrapper
                @tool(mcp_tool.name, description=mcp_tool.description)
                def mcp_tool_wrapper(**kwargs):
                    """Wrapper function that calls the MCP tool"""
                    logger.info(f"Calling MCP tool {mcp_tool.name} with args: {kwargs}")
                    try:
                        result = self.mcp_client.call_tool(
                            server_name="monkey-resonance-mcp",
                            name=mcp_tool.name,
                            arguments=kwargs
                        )
                        logger.info(f"MCP tool {mcp_tool.name} returned: {result}")
                        return result
                    except Exception as e:
                        logger.error(f"Error calling MCP tool {mcp_tool.name}: {e}", exc_info=True)
                        raise

                langchain_tools.append(mcp_tool_wrapper)
                logger.info(f"Created LangChain tool wrapper for {mcp_tool.name}")

            return langchain_tools
        except Exception as e:
            logger.error(f"Error getting tools from MCP: {e}", exc_info=True)
            return []
    
    def _convert_messages(self, messages):
        logger.info(f"Converting {len(messages)} messages to LangChain format")
        history = []
        for i, msg in enumerate(messages[:-1]):
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
                logger.info(f"Converted message {i} to HumanMessage")
            elif msg["role"] in ("assistant", "ai"):
                history.append(AIMessage(content=msg["content"]))
                logger.info(f"Converted message {i} to AIMessage")
        logger.info(f"Converted {len(history)} messages to chat history")
        return history


    def get_model(self) -> Dict[str, Any]:
        """Return model information for OpenWebUI"""
        logger.info("get_model called")
        return {
            "model": self.valves.MODEL,
            "label": self.name,
            "type": "pipe",
            "description": "Audio equipment manager agent that controls sound cards, applies effects, manages volumes, and handles recordings."
        }

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return the tools available to this pipeline"""
        logger.info("get_tools called")
        # Get fresh tools for the response
        tools = self._get_tools()
        logger.info(f"Returning {len(tools)} tools")
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args
            }
            for tool in tools
        ]
    




