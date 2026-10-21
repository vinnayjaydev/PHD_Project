Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Starting FLAME-AI Financial Lakehouse & Agentic Intelligence Dashboard" -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Cyan
python -m streamlit run ui/app.py --server.port 8501 --server.headless false
