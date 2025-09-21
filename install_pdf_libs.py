#!/usr/bin/env python3
"""
Install PDF parsing libraries automatically
"""
import subprocess
import sys

def install_pdf_libraries():
    """Install PDF parsing libraries"""
    print("📄 Installing PDF parsing libraries...")
    
    libraries = [
        "PyPDF2",
        "pdfplumber", 
        "pymupdf"
    ]
    
    for lib in libraries:
        try:
            print(f"Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"✅ {lib} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {lib}: {e}")
            return False
    
    return True

def test_imports():
    """Test that all libraries can be imported"""
    print("\n🧪 Testing imports...")
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError:
        print("❌ PyPDF2 import failed")
        return False
    
    try:
        import pdfplumber
        print("✅ pdfplumber imported successfully")
    except ImportError:
        print("❌ pdfplumber import failed")
        return False
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF imported successfully")
    except ImportError:
        print("❌ PyMuPDF import failed")
        return False
    
    return True

def main():
    """Main installation function"""
    print("🚀 PDF Libraries Installation Script")
    print("=" * 40)
    
    # Install libraries
    if not install_pdf_libraries():
        print("❌ Installation failed")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    print("\n🎉 All PDF libraries installed and working!")
    print("You can now use PDF parsing in the Deep Researcher Agent.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
