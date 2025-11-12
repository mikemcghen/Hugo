@echo off
REM Hugo Quick Start Script for Windows
REM ====================================

echo.
echo ================================================
echo Hugo Local AI Assistant - Quick Start
echo ================================================
echo.

REM Check if Ollama is running
echo [1/3] Checking Ollama connection...
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama is not running!
    echo.
    echo Please start Ollama first:
    echo   1. Open a new terminal
    echo   2. Run: ollama serve
    echo   3. Then run this script again
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama is running

REM Check if model is available
echo.
echo [2/3] Checking for llama3:8b model...
ollama list | findstr "llama3:8b" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Model not found. Pulling llama3:8b...
    ollama pull llama3:8b
    if %errorlevel% neq 0 (
        echo [!] Failed to pull model
        pause
        exit /b 1
    )
)
echo [OK] Model available

REM Start Hugo
echo.
echo [3/3] Starting Hugo interactive shell...
echo.
echo ================================================
echo.
python -m runtime.cli shell

echo.
echo Hugo session ended.
pause
