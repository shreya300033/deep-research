# 🌐 Custom Domain Setup Guide

## 🎯 Getting Your Deep Researcher Agent on a .com Domain

### **Step 1: Deploy Your App to Cloud**

#### **Option A: Streamlit Cloud (Recommended - Free)**
1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Deploy Deep Researcher Agent"
   git remote add origin https://github.com/yourusername/deep-researcher-agent.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Click "Deploy"
   - You'll get: `https://your-app-name.streamlit.app`

#### **Option B: Railway (Also Free)**
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub account
3. Select your repository
4. Deploy automatically
5. You'll get: `https://your-app-name.up.railway.app`

### **Step 2: Buy a Custom Domain**

#### **Recommended Domain Registrars:**
- **Namecheap**: $8.88/year (.com)
- **Cloudflare**: $9.77/year (.com)
- **Google Domains**: $12/year (.com)
- **GoDaddy**: $12.99/year (.com)

#### **Domain Name Ideas:**
- `deepresearcher.com`
- `airesearchagent.com`
- `smartresearcher.com`
- `airesearchhub.com`
- `intelligentresearch.com`

### **Step 3: Connect Domain to Your App**

#### **For Streamlit Cloud:**
1. **In Streamlit Cloud Dashboard:**
   - Go to your app settings
   - Click "Custom domain"
   - Enter your domain name
   - Follow DNS setup instructions

2. **DNS Configuration:**
   ```
   Type: CNAME
   Name: www (or @)
   Value: your-app-name.streamlit.app
   TTL: 3600
   ```

#### **For Railway:**
1. **In Railway Dashboard:**
   - Go to your project
   - Click "Settings" → "Domains"
   - Add your custom domain
   - Follow DNS instructions

2. **DNS Configuration:**
   ```
   Type: CNAME
   Name: www (or @)
   Value: your-app-name.up.railway.app
   TTL: 3600
   ```

#### **For Heroku:**
1. **In Heroku Dashboard:**
   - Go to your app
   - Click "Settings" → "Domains"
   - Add your custom domain
   - Follow DNS instructions

2. **DNS Configuration:**
   ```
   Type: CNAME
   Name: www
   Value: your-app-name.herokuapp.com
   TTL: 3600
   ```

### **Step 4: SSL Certificate (Automatic)**

Most platforms provide free SSL certificates:
- **Streamlit Cloud**: Automatic HTTPS
- **Railway**: Automatic HTTPS
- **Heroku**: Automatic HTTPS

### **Step 5: Final Result**

After setup, your app will be accessible at:
- **Primary:** `https://yourdomain.com`
- **WWW:** `https://www.yourdomain.com`

## 🚀 **Quick Start Commands**

### **Deploy to Streamlit Cloud:**
```bash
# 1. Initialize Git
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "Deploy Deep Researcher Agent"

# 4. Add remote (replace with your GitHub repo)
git remote add origin https://github.com/yourusername/deep-researcher-agent.git

# 5. Push to GitHub
git push -u origin main

# 6. Go to share.streamlit.io and deploy
```

### **Expected Timeline:**
- **Deployment**: 5-10 minutes
- **Domain Setup**: 10-15 minutes
- **DNS Propagation**: 1-24 hours
- **Total**: Same day live!

## 💰 **Cost Breakdown**

### **Free Option:**
- **Hosting**: Free (Streamlit Cloud/Railway)
- **Domain**: $8-12/year
- **SSL**: Free
- **Total**: ~$10/year

### **Premium Option:**
- **Hosting**: $5-20/month (Heroku/Railway Pro)
- **Domain**: $8-12/year
- **SSL**: Free
- **Total**: ~$70-250/year

## 🎯 **Next Steps After Deployment**

1. **Test your app** on the public URL
2. **Set up monitoring** and analytics
3. **Configure backups** and updates
4. **Add user authentication** (if needed)
5. **Set up email notifications** for errors

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **DNS not working**: Wait 24 hours for propagation
2. **SSL issues**: Check domain configuration
3. **App not loading**: Verify deployment logs
4. **Custom domain not working**: Check CNAME records

### **Support:**
- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io)
- **Railway**: [docs.railway.app](https://docs.railway.app)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)

---

**Your Deep Researcher Agent will be live at yourdomain.com! 🚀**
