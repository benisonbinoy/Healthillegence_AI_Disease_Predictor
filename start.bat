@echo off
title MedX - Flask API (port 5000)
color 0A

echo ============================================================
echo   MEDX - Flask API Server
echo ============================================================
echo.

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5000 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)

cd /d "%~dp0backend"
echo Starting Flask API on http://localhost:5000 ...
echo.
"C:\Program Files\Python311\python.exe" api_server.py
pause