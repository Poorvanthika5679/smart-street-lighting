@echo off
echo ============================================================
echo SMART STREET LIGHTING & ENERGY USAGE ANALYTICS SYSTEM
echo Launching Streamlit Hackathon Prototype...
echo ============================================================
echo.

echo [1/3] Checking dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Verifying sample dataset...
python generate_data.py

echo.
echo [3/3] Starting Streamlit Server...
streamlit run app.py

pause
