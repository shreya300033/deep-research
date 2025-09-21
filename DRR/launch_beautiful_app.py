#!/usr/bin/env python3
"""
Launch script for the beautiful Deep Researcher Agent Streamlit app
"""
import subprocess
import sys
import os

def launch_beautiful_app():
    """Launch the enhanced Streamlit app"""
    print("🎨 Launching Beautiful Deep Researcher Agent Interface")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ Error: streamlit_app.py not found in current directory")
        print("Please run this script from the DRR project directory")
        return False
    
    # Check if virtual environment exists
    if not os.path.exists(".venv"):
        print("❌ Error: Virtual environment not found")
        print("Please run setup.py first to create the virtual environment")
        return False
    
    print("🚀 Starting the beautiful Streamlit interface...")
    print("📱 The app will open in your default browser")
    print("🎨 Enjoy the enhanced UI with modern styling!")
    print("=" * 60)
    
    try:
        # Launch Streamlit with the enhanced app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
        return True

def main():
    """Main launcher function"""
    print("🔬 Deep Researcher Agent - Beautiful Interface Launcher")
    print("=" * 60)
    
    success = launch_beautiful_app()
    
    if success:
        print("✅ App launched successfully!")
    else:
        print("❌ Failed to launch app")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
