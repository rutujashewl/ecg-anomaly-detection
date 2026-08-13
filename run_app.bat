@echo off
cd /d "%~dp0"
start http://localhost:8501
py -3.11 -m streamlit run app.py