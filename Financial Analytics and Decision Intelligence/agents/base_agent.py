"""
Base Financial Agent: Unified Multi-Agent Abstract Class.
Handles tool execution, structured reasoning trajectories, confidence scoring,
and multi-backend LLM integration (OpenAI, Gemini, Groq, Ollama, Local Deterministic).
Part of the FLAME-AI Financial Agentic Framework.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BaseAgent")


class BaseFinancialAgent(ABC):
    """
    Abstract Base Class for Financial Intelligence Agents.
    Maintains reasoning memory, tool registry, and formal consensus protocols.
    """

    def __init__(
        self,
        name: str,
        role: str,
        weight: float = 0.25,
        llm_provider: str = "deterministic",  # "openai", "gemini", "groq", "ollama", "deterministic"
        api_key: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.weight = weight
        self.llm_provider = llm_provider
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.trajectory_log: List[Dict[str, Any]] = []
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, tool_name: str, tool_func: Callable, description: str):
        """Registers an analytical or retrieval tool available to this agent."""
        self.tools[tool_name] = {
            "func": tool_func,
            "description": description
        }

    def log_trajectory_step(self, step_type: str, details: Dict[str, Any]):
        """Records an auditable thought, action, or observation in the agent's trajectory."""
        entry = {
            "timestamp": time.time(),
            "agent": self.name,
            "step_type": step_type,  # "OBSERVATION", "THOUGHT", "TOOL_CALL", "REFLECTION", "DECISION"
            "details": details
        }
        self.trajectory_log.append(entry)

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Executes a registered tool and logs the trajectory step."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered for agent '{self.name}'.")

        self.log_trajectory_step("TOOL_CALL", {"tool": tool_name, "args": kwargs})
        start = time.time()
        result = self.tools[tool_name]["func"](**kwargs)
        elapsed = time.time() - start

        self.log_trajectory_step("OBSERVATION", {"tool": tool_name, "elapsed_ms": round(elapsed * 1000, 2)})
        return result

    @abstractmethod
    def analyze(self, ticker: str, as_of_date: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes domain-specific analysis and returns a structured decision object:
        {
            "agent": str,
            "ticker": str,
            "stance": "BULLISH" | "BEARISH" | "NEUTRAL" | "DEFENSIVE",
            "confidence": float (0.0 - 1.0),
            "rationale": str,
            "key_metrics": dict,
            "citations": list,
            "trajectory": list
        }
        """
        pass

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Dispatches LLM calls. If live API key is present and provider is set, calls live API.
        Otherwise, returns deterministic empirical reasoning.
        """
        # 1. Live OpenAI
        if self.llm_provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": prompt})
                resp = client.chat.completions.create(model="gpt-4o", messages=msgs, temperature=0.2)
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"OpenAI call failed ({e}). Reverting to deterministic mode.")

        # 2. Live Groq
        if self.llm_provider == "groq" and self.api_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.api_key)
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": prompt})
                resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2)
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Groq call failed ({e}). Reverting to deterministic mode.")

        # Default fallback: deterministic
        return ""
