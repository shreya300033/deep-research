@echo off
echo Starting Deep Researcher Agent - Streamlit Interface
echo ==================================================

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Error: Virtual environment not activated
    pause
    exit /b 1
)

echo Virtual environment activated: %VIRTUAL_ENV%

REM Install streamlit if not already installed
echo Checking Streamlit installation...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing Streamlit...
    pip install streamlit
)

REM Launch Streamlit
echo Launching Streamlit interface...
echo The web interface will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py --server.port 8501 --server.address localhost

pause
