# 🚀 Deep Researcher Agent - Deployment Guide

This guide provides multiple deployment options for the Deep Researcher Agent, from local development to cloud production deployment.

## 📋 Prerequisites

- Python 3.9+ installed
- Git installed
- Docker (optional, for containerized deployment)
- Cloud platform account (Heroku, Railway, Streamlit Cloud, etc.)

## 🛠️ Quick Setup

Run the deployment setup script:
```bash
python deploy.py
```

This will:
- ✅ Check all required files
- ✅ Create necessary directories
- ✅ Set up configuration files
- ✅ Test the application structure

## 🌐 Deployment Options

### 1. 🐳 Docker Deployment (Recommended for Production)

#### Local Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t deep-researcher-agent .
docker run -p 8501:8501 deep-researcher-agent
```

#### Deploy to Cloud with Docker
- **Railway**: Connect GitHub repo with Dockerfile
- **Heroku**: Use container stack with Dockerfile
- **AWS/GCP/Azure**: Use container services

### 2. ☁️ Streamlit Cloud (Easiest)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/deep-researcher-agent.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the repository and main branch
   - Click "Deploy"

### 3. 🚀 Heroku Deployment

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create Heroku app**:
   ```bash
   heroku create your-app-name
   ```

3. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

4. **Open your app**:
   ```bash
   heroku open
   ```

### 4. 🚂 Railway Deployment

1. **Connect GitHub**:
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub account
   - Select your repository

2. **Deploy**:
   - Railway will automatically detect the `railway.json` configuration
   - The app will deploy automatically on push

### 5. 🌐 Local Development

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for local development:
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Streamlit Configuration

The `streamlit_config.toml` file contains production-ready settings:
- Headless mode enabled
- CORS disabled for cloud deployment
- Custom theme colors
- Optimized server settings

## 🔧 Production Optimizations

### Memory and Performance
- The app uses FAISS for efficient vector search
- Embeddings are cached for better performance
- PDF processing is optimized for large files

### Security
- XSRF protection can be enabled in production
- CORS settings can be configured as needed
- Environment variables for sensitive data

## 📊 Monitoring and Logs

### Health Checks
- Docker: Built-in health check endpoint
- Heroku: Automatic health monitoring
- Railway: Health check configuration included

### Logs
```bash
# Docker
docker-compose logs -f

# Heroku
heroku logs --tail

# Railway
railway logs
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Issues**:
   - Ensure port 8501 is available
   - Check firewall settings
   - Use environment variable `PORT` for cloud platforms

2. **Memory Issues**:
   - Increase memory limits in cloud platform
   - Optimize batch processing
   - Use smaller embedding models

3. **Dependency Issues**:
   - Ensure Python 3.9+ is used
   - Check all requirements are installed
   - Verify virtual environment is activated

### Debug Mode
```bash
# Run with debug information
streamlit run streamlit_app.py --logger.level=debug
```

## 📈 Scaling Considerations

### For High Traffic
- Use multiple instances behind a load balancer
- Implement Redis for session storage
- Consider using GPU instances for faster embeddings

### For Large Datasets
- Use distributed vector stores (Pinecone, Weaviate)
- Implement batch processing
- Add data persistence layers

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Streamlit Cloud
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit Cloud
      # Add your deployment steps here
```

## 📞 Support

If you encounter issues:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure the virtual environment is activated
4. Check cloud platform documentation

## 🎯 Next Steps

After successful deployment:
1. Test all features in production
2. Set up monitoring and alerts
3. Configure custom domain (if needed)
4. Implement user authentication (if required)
5. Add backup and recovery procedures

---

**Happy Deploying! 🚀**
