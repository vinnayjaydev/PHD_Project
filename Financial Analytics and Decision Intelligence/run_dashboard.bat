@echo off
echo =======================================================================
echo Starting FLAME-AI Financial Lakehouse & Agentic Intelligence Dashboard
echo =======================================================================
python -m streamlit run ui/app.py --server.port 8501 --server.headless false
pause
