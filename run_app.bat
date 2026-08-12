@echo off
echo Starting ECG Anomaly Detection App...
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run app.py
pause
