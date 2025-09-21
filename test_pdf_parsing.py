#!/usr/bin/env python3
"""
Test script for PDF parsing functionality
"""
import sys
import os

def test_pdf_imports():
    """Test PDF parsing library imports"""
    print("🧪 Testing PDF parsing library imports...")
    
    libraries = {
        'PyPDF2': False,
        'pdfplumber': False,
        'pymupdf': False
    }
    
    try:
        import PyPDF2
        libraries['PyPDF2'] = True
        print("✅ PyPDF2 imported successfully")
    except ImportError:
        print("❌ PyPDF2 not available")
    
    try:
        import pdfplumber
        libraries['pdfplumber'] = True
        print("✅ pdfplumber imported successfully")
    except ImportError:
        print("❌ pdfplumber not available")
    
    try:
        import fitz  # PyMuPDF
        libraries['pymupdf'] = True
        print("✅ PyMuPDF (fitz) imported successfully")
    except ImportError:
        print("❌ PyMuPDF not available")
    
    return libraries

def test_pdf_parser():
    """Test the PDF parser utility"""
    print("\n🔧 Testing PDF parser utility...")
    
    try:
        from utils.pdf_parser import PDFParser
        parser = PDFParser()
        print(f"✅ PDFParser created successfully")
        print(f"📚 Available parsers: {parser.available_parsers}")
        
        if not parser.available_parsers:
            print("⚠️  No PDF parsers available. Install PyPDF2, pdfplumber, or pymupdf")
            return False
        
        return True
    except Exception as e:
        print(f"❌ PDFParser test failed: {e}")
        return False

def test_document_processor():
    """Test document processor with PDF support"""
    print("\n📄 Testing document processor...")
    
    try:
        from utils.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        
        # Test that PDF methods exist
        if hasattr(DocumentProcessor, 'process_pdf_file'):
            print("✅ process_pdf_file method available")
        else:
            print("❌ process_pdf_file method missing")
            return False
        
        if hasattr(DocumentProcessor, 'process_pdf_data'):
            print("✅ process_pdf_data method available")
        else:
            print("❌ process_pdf_data method missing")
            return False
        
        return True
    except Exception as e:
        print(f"❌ DocumentProcessor test failed: {e}")
        return False

def main():
    """Run all PDF parsing tests"""
    print("📄 PDF Parsing Test Suite")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test library imports
    libraries = test_pdf_imports()
    if not any(libraries.values()):
        print("\n❌ No PDF parsing libraries available!")
        print("Please install at least one of:")
        print("  - pip install PyPDF2")
        print("  - pip install pdfplumber")
        print("  - pip install pymupdf")
        all_tests_passed = False
    
    # Test PDF parser
    if not test_pdf_parser():
        all_tests_passed = False
    
    # Test document processor
    if not test_document_processor():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("🎉 All PDF parsing tests passed!")
        print("\nPDF parsing is ready to use in the Streamlit interface.")
        print("You can now upload PDF files in the Document Management tab.")
    else:
        print("❌ Some PDF parsing tests failed.")
        print("Please install the required libraries and try again.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
