from pydantic import BaseModel
from typing import Any, Dict, Generator

from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
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

        # Tools + prompt
        mcp_tools = self.mcp_client.get_tools()
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in mcp_tools
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are an MCP agent connected to tools at {self.valves.MCP_URL}.

Available tools:
{tool_descriptions}

Always use tools when external actions or data are required."""
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(
            self.llm,
            mcp_tools,
            self.prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=mcp_tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def _convert_messages(self, messages):
        history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] in ("assistant", "ai"):
                history.append(AIMessage(content=msg["content"]))
        return history

    def pipe(self, body: Dict[str, Any]) -> Generator[str, None, None]:
        messages = body.get("messages", [])
        if not messages:
            yield "No messages provided"
            return

        chat_history = self._convert_messages(messages)
        user_input = messages[-1]["content"]

        try:
            result = self.executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            yield result["output"]

        except Exception as e:
            yield f"MCP Agent Error: {str(e)}"
