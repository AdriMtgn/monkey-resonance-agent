from pipelines import Pipe
from pydantic import BaseModel
from typing import Any, Dict, Generator, Iterator, Union, AsyncGenerator
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

class Valves(BaseModel):
    """Pipe configuration options"""
    MCP_URL: str = "http://192.168.1.48:8000"
    OLLAMA_URL: str = "http://ollama.local"
    MODEL: str = "llama3.1"
    TEMPERATURE: float = 0.1

class MCPAgentPipe(Pipe):
    def __init__(self):
        super().__init__()
        self.valves = Valves()
        self.type = "pipe"
        self.name = "ollama-mcp-agent"
        
        # MCP Client
        self.mcp_client = MultiServerMCPClient({
            "monkey-resonance-mcp": {
                "url": self.valves.MCP_URL,
                "transport": "streamable_http"
            }
        })
        
        # Ollama LLM
        self.llm = ChatOllama(
            model=self.valves.MODEL,
            base_url=self.valves.OLLAMA_URL,
            temperature=self.valves.TEMPERATURE
        )
        
        # MCP-specific prompt with tool awareness
        mcp_tools = self.mcp_client.get_tools()
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an MCP agent connected to external tools at {self.valves.MCP_URL}.
            
Available MCP tools:
{tool_descriptions}

ALWAYS use MCP tools for external data/actions. Think step-by-step before calling tools.
Format tool calls correctly and explain your reasoning."""),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        self.agent = create_tool_calling_agent(self.llm, mcp_tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=mcp_tools, 
            verbose=True,
            handle_parsing_errors=True
        )

    def convert_messages(self, messages: list) -> list:
        """Convert OpenWebUI messages to LangChain format"""
        chat_history = []
        for msg in messages[:-1]:  # All but last message = history
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] in ["assistant", "ai"]:
                chat_history.append(AIMessage(content=msg["content"]))
        
        return chat_history

    def pipe(self, body: Dict[str, Any]) -> Generator[str, None, None]:
        """Main pipe method - handles full conversation history"""
        messages = body.get("messages", [])
        if not messages:
            yield "No messages provided"
            return
            
        # Convert full history for proper agent context
        chat_history = self.convert_messages(messages)
        user_input = messages[-1]["content"]
        
        try:
            # Full agent input with MCP context
            agent_input = {
                "input": user_input,
                "chat_history": chat_history
            }
            
            # Execute with streaming support
            result = self.executor.invoke(agent_input)
            yield result["output"]
            
        except Exception as e:
            yield f"MCP Agent Error: {str(e)}"
