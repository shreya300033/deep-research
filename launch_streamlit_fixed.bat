@echo off
echo 🎨 Launching Beautiful Deep Researcher Agent Interface
echo ==================================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Error: Virtual environment not found
    echo Please run setup.py first to create the virtual environment
    pause
    exit /b 1
)

echo 🚀 Activating virtual environment...
call .venv\Scripts\activate.bat

echo 📱 Starting Streamlit with beautiful UI...
echo 🎨 The app will open in your default browser
echo ==================================================

REM Launch Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
