#!/usr/bin/env python3
"""
Quick deployment script for Deep Researcher Agent to cloud platforms
"""
import os
import subprocess
import sys
from pathlib import Path

def check_git_status():
    """Check if git is initialized and files are ready"""
    if not Path('.git').exists():
        print("❌ Git not initialized. Initializing...")
        subprocess.run(['git', 'init'], check=True)
        print("✅ Git initialized")
    
    # Check if there are uncommitted changes
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    if result.stdout.strip():
        print("📝 Uncommitted changes found:")
        print(result.stdout)
        return False
    return True

def setup_git_repo():
    """Set up git repository for deployment"""
    print("🔧 Setting up Git repository...")
    
    # Add all files
    subprocess.run(['git', 'add', '.'], check=True)
    print("✅ Files added to git")
    
    # Commit
    subprocess.run(['git', 'commit', '-m', 'Deploy Deep Researcher Agent to cloud'], check=True)
    print("✅ Changes committed")
    
    print("\n📋 Next steps:")
    print("1. Create a GitHub repository at https://github.com/new")
    print("2. Add the remote origin:")
    print("   git remote add origin https://github.com/yourusername/deep-researcher-agent.git")
    print("3. Push to GitHub:")
    print("   git push -u origin main")
    print("4. Deploy on Streamlit Cloud:")
    print("   Go to https://share.streamlit.io and connect your repo")

def create_deployment_files():
    """Create additional files needed for cloud deployment"""
    print("📁 Creating deployment files...")
    
    # Create .streamlit/secrets.toml if it doesn't exist
    streamlit_dir = Path('.streamlit')
    streamlit_dir.mkdir(exist_ok=True)
    
    secrets_file = streamlit_dir / 'secrets.toml'
    if not secrets_file.exists():
        secrets_content = """# Streamlit secrets for cloud deployment
# Add any sensitive configuration here

[general]
# Add any general configuration
"""
        secrets_file.write_text(secrets_content)
        print("✅ Created .streamlit/secrets.toml")
    
    # Create .gitignore if it doesn't exist
    gitignore_file = Path('.gitignore')
    if not gitignore_file.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
"""
        gitignore_file.write_text(gitignore_content)
        print("✅ Created .gitignore")

def main():
    """Main deployment setup function"""
    print("🚀 Deep Researcher Agent - Cloud Deployment Setup")
    print("=" * 60)
    
    # Create deployment files
    create_deployment_files()
    
    # Check git status
    if not check_git_status():
        setup_git_repo()
    else:
        print("✅ Git repository is ready")
    
    print("\n🎯 Deployment Options:")
    print("1. 🌟 Streamlit Cloud (Recommended - Free)")
    print("   - Go to https://share.streamlit.io")
    print("   - Connect your GitHub repository")
    print("   - Deploy automatically")
    print("   - Get: https://your-app-name.streamlit.app")
    
    print("\n2. 🚂 Railway (Also Free)")
    print("   - Go to https://railway.app")
    print("   - Connect GitHub repository")
    print("   - Deploy automatically")
    print("   - Get: https://your-app-name.up.railway.app")
    
    print("\n3. 🚀 Heroku")
    print("   - Install Heroku CLI")
    print("   - Create app: heroku create your-app-name")
    print("   - Deploy: git push heroku main")
    print("   - Get: https://your-app-name.herokuapp.com")
    
    print("\n🌐 Custom Domain Setup:")
    print("1. Buy domain from Namecheap/GoDaddy ($8-12/year)")
    print("2. Add CNAME record pointing to your app URL")
    print("3. Configure in your platform's domain settings")
    print("4. Get: https://yourdomain.com")
    
    print("\n📖 See CUSTOM_DOMAIN_SETUP.md for detailed instructions")
    
    print("\n✅ Your app is ready for cloud deployment!")
    print("🎯 Expected result: https://yourdomain.com")

if __name__ == "__main__":
    main()
