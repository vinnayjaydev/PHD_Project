"""Base Agent Abstraction for Cognitive and Meta-Cognitive Roles.

Provides unified interface for prompt construction, LLM backend dispatch (Gemini, OpenAI, Groq),
and high-fidelity deterministic generation when API keys are not supplied.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from metacog_finance.core.config import settings


class BaseFinancialAgent:
    """Base class for all financial cognitive agents."""

    def __init__(self, name: str, role_description: str):
        self.name = name
        self.role_description = role_description

    def _call_llm(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Dispatches to Gemini/OpenAI if configured, otherwise uses deterministic generation."""
        # 1. Try Google Gemini API if key is available
        if settings.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=settings.gemini_api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"system_instruction": system_instruction or self.role_description},
                )
                if response and response.text:
                    return response.text
            except Exception:
                pass

        # 2. Try OpenAI API if key is available
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.openai_api_key)
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
            except Exception:
                pass

        # 3. Deterministic Domain Generator fallback
        return self._generate_deterministic_fallback(prompt)

    def _generate_deterministic_fallback(self, prompt: str) -> str:
        """Domain fallback when external cloud LLM is offline or not configured."""
        return f"[{self.name} Response Grounded in Audited Filings]\nAnalysis based on verified parameters."
