#!/usr/bin/env python3
"""
Launcher script for the Streamlit Deep Researcher Agent interface
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    print("🚀 Starting Deep Researcher Agent - Streamlit Interface")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the DRR directory.")
        return
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not installed!")
        print("Installing Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            return
    
    print("\n🌐 Launching Streamlit interface...")
    print("📱 The web interface will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Click 'Initialize Agent' in the sidebar to start")
    print("   - Load sample documents to get started quickly")
    print("   - Use the Research Query tab to ask questions")
    print("   - Check Analytics for research insights")
    print("   - Export reports in PDF or Markdown format")
    print("\n" + "=" * 60)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface closed. Goodbye!")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

if __name__ == "__main__":
    main()
