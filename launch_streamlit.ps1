# PowerShell script to launch Streamlit interface
Write-Host "🚀 Starting Deep Researcher Agent - Streamlit Interface" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "❌ Error: Virtual environment not activated" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green

# Check Streamlit installation
Write-Host "Checking Streamlit installation..." -ForegroundColor Yellow
try {
    python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
    Write-Host "✅ Streamlit is installed" -ForegroundColor Green
} catch {
    Write-Host "Installing Streamlit..." -ForegroundColor Yellow
    pip install streamlit
}

# Launch Streamlit
Write-Host "🌐 Launching Streamlit interface..." -ForegroundColor Green
Write-Host "📱 The web interface will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "🔗 If it doesn't open automatically, go to: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   - Click 'Initialize Agent' in the sidebar to start" -ForegroundColor White
Write-Host "   - Load sample documents to get started quickly" -ForegroundColor White
Write-Host "   - Use the Research Query tab to ask questions" -ForegroundColor White
Write-Host "   - Check Analytics for research insights" -ForegroundColor White
Write-Host "   - Export reports in PDF or Markdown format" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host "==================================================" -ForegroundColor Green

# Launch Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
