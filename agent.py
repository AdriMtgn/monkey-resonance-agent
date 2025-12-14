from pydantic import BaseModel
from typing import Any, Dict, Generator

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

        tools = self.mcp_client.get_tools()

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

        # 🔑 LCEL tool binding
        self.chain = (
            prompt
            | llm.bind_tools(tools)
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

        result = self.chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        yield result.content