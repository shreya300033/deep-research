"""
Setup script for the Deep Researcher Agent
"""
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating necessary directories...")
    
    directories = ["data", "embeddings", "index", "reports"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def test_installation():
    """Test the installation"""
    print("\nTesting installation...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Installation test passed!")
            return True
        else:
            print("❌ Installation test failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Installation test timed out!")
        return False
    except Exception as e:
        print(f"❌ Installation test error: {e}")
        return False

def main():
    """Main setup function"""
    print("=== Deep Researcher Agent Setup ===\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("\n⚠️  Installation test failed, but you can still try running the system manually.")
        print("Run 'python test_system.py' to diagnose issues.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python example_usage.py' to see the full demo")
    print("2. Check the README.md for detailed usage instructions")
    print("3. Start your own research by modifying example_usage.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
