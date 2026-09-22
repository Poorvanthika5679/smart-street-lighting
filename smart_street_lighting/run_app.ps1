Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SMART STREET LIGHTING & ENERGY USAGE ANALYTICS SYSTEM" -ForegroundColor Green
Write-Host "Launching Streamlit Hackathon Prototype..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "[2/3] Verifying sample dataset..." -ForegroundColor Yellow
python generate_data.py

Write-Host ""
Write-Host "[3/3] Starting Streamlit Server..." -ForegroundColor Green
streamlit run app.py
