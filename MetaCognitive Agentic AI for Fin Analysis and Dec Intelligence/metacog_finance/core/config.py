"""Configuration parameters and settings for the Meta-Cognitive Financial Framework."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class MetaCogSettings(BaseSettings):
    """Runtime configuration settings."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Keys (optional for mock/offline mode)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # SEC EDGAR requirements (SEC requires User-Agent header: SampleCompany AdminContact@<sample.com>)
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "MetaCognitiveResearchLab research@agenticfin.org")

    # Meta-Cognitive Calibration Hyperparameters
    # Formula: Calibrated_Confidence = alpha * Faithfulness + beta * NarrativeConsistency + gamma * VerificationPass
    calib_weight_faithfulness: float = 0.40
    calib_weight_consistency: float = 0.35
    calib_weight_verification: float = 0.25

    # Discrepancy penalty per severity
    penalty_critical: float = 0.35
    penalty_high: float = 0.20
    penalty_medium: float = 0.10
    penalty_low: float = 0.05

    # Backtracking thresholds
    confidence_backtrack_threshold: float = 0.70
    max_backtrack_iterations: int = 2

    # Storage paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    cache_dir: Path = base_dir / "cache"
    demo_data_dir: Path = base_dir / "demo_data"


settings = MetaCogSettings()
# Ensure directories exist
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.demo_data_dir.mkdir(parents=True, exist_ok=True)
