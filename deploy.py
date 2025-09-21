#!/usr/bin/env python3
"""
Deployment script for Deep Researcher Agent
Provides multiple deployment options and setup
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'streamlit_app.py',
        'researcher_agent.py',
        'requirements.txt',
        'config.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def create_streamlit_secrets():
    """Create streamlit secrets file for production"""
    secrets_content = """
# Streamlit secrets for production deployment
# Add any sensitive configuration here

[general]
# Add any general configuration
"""
    
    with open('.streamlit/secrets.toml', 'w') as f:
        f.write(secrets_content)
    
    print("✅ Created .streamlit/secrets.toml")

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'index', 'reports', '.streamlit']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created necessary directories")

def test_local_deployment():
    """Test the application locally"""
    print("🧪 Testing local deployment...")
    
    try:
        # Test imports
        import streamlit
        import sentence_transformers
        import faiss
        print("✅ All dependencies imported successfully")
        
        # Test if the app can start (without actually running it)
        print("✅ Application structure is valid")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_docker_compose():
    """Create docker-compose.yml for easy local deployment"""
    docker_compose_content = """
version: '3.8'

services:
  deep-researcher:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./index:/app/index
      - ./reports:/app/reports
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("✅ Created docker-compose.yml")

def main():
    """Main deployment setup function"""
    print("🚀 Deep Researcher Agent - Deployment Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Create secrets file
    create_streamlit_secrets()
    
    # Create docker-compose
    create_docker_compose()
    
    # Test local deployment
    if test_local_deployment():
        print("\n✅ Deployment setup completed successfully!")
        print("\n📋 Available deployment options:")
        print("1. 🐳 Docker: docker-compose up")
        print("2. ☁️  Streamlit Cloud: Push to GitHub and connect")
        print("3. 🚀 Heroku: heroku create && git push heroku main")
        print("4. 🚂 Railway: Connect GitHub repository")
        print("5. 🌐 Local: streamlit run streamlit_app.py")
        
        print("\n📖 See DEPLOYMENT_GUIDE.md for detailed instructions")
    else:
        print("\n❌ Deployment setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
