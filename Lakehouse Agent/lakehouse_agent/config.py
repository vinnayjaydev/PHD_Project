"""
System Configuration & Environment Setup
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LAKEHOUSE_DIR = DATA_DIR / "lakehouse"
BRONZE_DIR = LAKEHOUSE_DIR / "bronze"
SILVER_DIR = LAKEHOUSE_DIR / "silver"
GOLD_DIR = LAKEHOUSE_DIR / "gold"
VECTOR_DIR = DATA_DIR / "vector_store"
DOCS_DIR = DATA_DIR / "raw_documents"

# Ensure all directories exist
for path in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, VECTOR_DIR, DOCS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Lakehouse Settings
    lakehouse_db_path: str = str(LAKEHOUSE_DIR / "financial_lakehouse.duckdb")
    bronze_path: Path = BRONZE_DIR
    silver_path: Path = SILVER_DIR
    gold_path: Path = GOLD_DIR
    vector_path: Path = VECTOR_DIR

    # Financial / Compliance Rules
    max_equity_allocation_conservative: float = 0.40
    max_equity_allocation_moderate: float = 0.70
    max_equity_allocation_aggressive: float = 0.90
    min_emergency_fund_months: int = 6
    max_single_stock_concentration: float = 0.15

    # Web Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
