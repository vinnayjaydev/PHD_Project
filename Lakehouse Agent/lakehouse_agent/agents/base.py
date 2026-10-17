"""
Base Agent Architecture
Provides structured thinking traces, tool execution logging, and unified LLM interfaces.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from lakehouse_agent.agents.llm_provider import LLMClient


class BaseAgent:
    """Abstract Base Agent for the financial intelligence system."""

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        system_prompt: str,
        llm_client: Optional[LLMClient] = None
    ):
        self.name = name
        self.role = role
        self.description = description
        self.system_prompt = system_prompt
        self.llm = llm_client or LLMClient()
        self.trace: List[Dict[str, Any]] = []

    def log_step(self, step_name: str, details: Any):
        """Records an execution step in the agent's chain-of-thought trace."""
        self.trace.append({
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "step": step_name,
            "details": details
        })

    def get_trace(self) -> List[Dict[str, Any]]:
        """Returns the complete execution trace for explainability."""
        return self.trace

    def clear_trace(self):
        """Resets the trace for new execution cycles."""
        self.trace = []

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Core execution method to be implemented by specialized agents."""
        raise NotImplementedError("Specialized agents must implement execute().")
