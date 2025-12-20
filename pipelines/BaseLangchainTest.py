"""
title: LangChain Weather Agent Pipeline (Local Ollama)
author: Assistant
date: 2025-12-20
version: 1.1
license: MIT
description: OpenWebUI Pipeline using LangChain agent with Ollama local LLM and weather tool.
requirements: langchain, langchain-ollama, pydantic
"""

from typing import List, Union, Generator
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

class Pipeline:
    class Valves(BaseModel):
        OLLAMA_URL: str = "http://test-chat_ollama:11434"
        MODEL: str = "llama2:latest"
        TEMPERATURE: float = 0.1

    def __init__(self):
        self.name = "LangChain Weather Agent (Ollama)"
        self.agent = None
        self.valves = self.Valves()

    async def on_startup(self):
        """Initialize the LangChain agent with Ollama on pipeline startup."""
        model = ChatOllama(
            model=self.valves.MODEL,
            base_url=self.valves.OLLAMA_URL,
            temperature=self.valves.TEMPERATURE
        )
        self.agent = create_agent(
            model=model,
            tools=[get_weather],
            system_prompt="You are a helpful assistant with access to a weather tool. Use the get_weather tool when asked about weather.",
        )
        print(f"LangChain Weather Agent initialized with Ollama model: {self.valves.MODEL}")

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Generator[str, None, None]:
        """Process messages through the LangChain agent."""
        # Convert OpenWebUI messages to LangChain format
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
        
        try:
            # Invoke agent and stream response
            for chunk in self.agent.stream({"messages": langchain_messages}):
                content = chunk["messages"][-1]["content"]
                if content:
                    yield content
                    
        except Exception as e:
            yield f"Error: {str(e)}. Ensure Ollama is running with model '{self.valves.MODEL}' at {self.valves.OLLAMA_URL}"
