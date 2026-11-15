@echo off
title CineVision AI - Streamlit Dashboard
cd /d "%~dp0"
echo ===================================================
echo   CineVision AI 2.0 - Streamlit Web Dashboard
echo ===================================================
echo.
echo Launching Streamlit web application on http://localhost:8501 ...
echo.
streamlit run app.py
pause

