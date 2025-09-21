#!/usr/bin/env python3
"""
Test script to verify Streamlit app components work
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing Streamlit app imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from researcher_agent import DeepResearcherAgent
        print("✅ DeepResearcherAgent imported successfully")
    except ImportError as e:
        print(f"❌ DeepResearcherAgent import failed: {e}")
        return False
    
    try:
        from utils.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
    except ImportError as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        return False
    
    return True

def test_agent_initialization():
    """Test that the agent can be initialized"""
    print("\n🤖 Testing agent initialization...")
    
    try:
        from researcher_agent import DeepResearcherAgent
        agent = DeepResearcherAgent()
        print("✅ Agent initialized successfully")
        
        session_id = agent.start_research_session("test")
        print(f"✅ Session started: {session_id}")
        
        return True
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def test_streamlit_app():
    """Test that the Streamlit app can be loaded"""
    print("\n🌐 Testing Streamlit app loading...")
    
    try:
        # Check if the app file exists
        if not os.path.exists("streamlit_app.py"):
            print("❌ streamlit_app.py not found")
            return False
        
        print("✅ streamlit_app.py found")
        
        # Try to compile the app (basic syntax check)
        with open("streamlit_app.py", "r") as f:
            code = f.read()
        
        compile(code, "streamlit_app.py", "exec")
        print("✅ streamlit_app.py syntax is valid")
        
        return True
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 Deep Researcher Agent - Streamlit Test Suite")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test agent initialization
    if not test_agent_initialization():
        all_tests_passed = False
    
    # Test Streamlit app
    if not test_streamlit_app():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Streamlit app should work correctly.")
        print("\nTo launch the app:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Or use: python launch_streamlit.ps1")
        print("3. Open: http://localhost:8501")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
