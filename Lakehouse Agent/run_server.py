"""
Launcher script for Lakehouse Agent Server & Interactive Web UI
"""
import sys
import uvicorn
from lakehouse_agent.config import settings

def main():
    print("=" * 70)
    print("  LAKEHOUSE-ENABLED AGENTIC INTELLIGENCE SYSTEM")
    print("  Adaptive Personalized Financial Recommendation with RAG")
    print("=" * 70)
    print(f"[*] Initializing embedded DuckDB Lakehouse at: {settings.lakehouse_db_path}")
    print(f"[*] Active LLM Provider: {settings.llm_provider.upper()}")
    print(f"[*] Starting Web Server on http://{settings.host}:{settings.port}")
    print("=" * 70)
    
    uvicorn.run(
        "lakehouse_agent.web.app:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )

if __name__ == "__main__":
    main()
