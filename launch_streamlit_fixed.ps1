# PowerShell script to launch Streamlit with proper environment
Write-Host "🎨 Launching Beautiful Deep Researcher Agent Interface" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "❌ Error: Virtual environment not found" -ForegroundColor Red
    Write-Host "Please run setup.py first to create the virtual environment" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🚀 Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

Write-Host "📱 Starting Streamlit with beautiful UI..." -ForegroundColor Green
Write-Host "🎨 The app will open in your default browser" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Launch Streamlit
& streamlit run streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

Read-Host "Press Enter to exit"
