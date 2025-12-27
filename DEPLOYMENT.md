# 🚀 Deployment Guide for Streamlit Cloud

This guide will help you deploy your Alzheimer's EEG Prediction app to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: Create one at [github.com](https://github.com) if you don't have one
2. **Streamlit Cloud Account**: Sign up at [streamlit.io/cloud](https://streamlit.io/cloud) using your GitHub account
3. **Git installed**: Download from [git-scm.com](https://git-scm.com/)

## 📦 Step 1: Prepare Your Repository

### Files Included for Deployment:

✅ **Essential Files** (already created):
- `.gitignore` - Excludes large data files from Git
- `.streamlit/config.toml` - Streamlit configuration
- `packages.txt` - System dependencies
- `requirements.txt` - Python dependencies
- `app.py` - Main Streamlit application
- `README.md` - Project documentation
- Model files (`.pkl` files) - Pre-trained models

⚠️ **Note**: Large data files (`.edf`, `.set`, `processed_features.csv`) are excluded from Git as they're too large for GitHub.

## 🔧 Step 2: Initialize Git Repository

Open PowerShell/Command Prompt in your project directory and run:

```powershell
# Navigate to project directory
cd "c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\csvcaueeg"

# Initialize Git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create first commit
git commit -m "Initial commit: Alzheimer's EEG Prediction App"
```

## 🌐 Step 3: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `alzheimers-eeg-prediction` (or your preferred name)
3. Description: "ML-powered Alzheimer's stage prediction from EEG data"
4. Choose **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Option B: Using GitHub CLI (if installed)

```powershell
gh repo create alzheimers-eeg-prediction --public --source=. --remote=origin
```

## 📤 Step 4: Push to GitHub

After creating the repository on GitHub, connect and push:

```powershell
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/alzheimers-eeg-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## ☁️ Step 5: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure deployment**:
   - **Repository**: Select `YOUR_USERNAME/alzheimers-eeg-prediction`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (e.g., `alzheimers-eeg-predictor`)

5. **Advanced settings** (optional):
   - Python version: 3.11 (recommended)
   - Secrets: Add any API keys if needed (not required for this app)

6. **Click "Deploy"**

7. **Wait for deployment** (usually 2-5 minutes)

## 🎉 Step 6: Access Your App

Once deployed, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

Share this URL with anyone to use your Alzheimer's prediction tool!

## 🔄 Updating Your App

To update your deployed app:

```powershell
# Make your changes to the code
# Then commit and push

git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically redeploy your app within minutes!

## ⚠️ Important Notes

### File Size Limitations

- **GitHub**: Max 100 MB per file, 1 GB per repository
- **Streamlit Cloud**: 1 GB total storage

Your model files should be fine, but large datasets need to be excluded (already done in `.gitignore`).

### Model Files

The following files **MUST** be in your repository for the app to work:
- ✅ `alzheimer_model.pkl` (~2.3 MB)
- ✅ `scaler.pkl` (~23 KB)
- ✅ `feature_selector.pkl` (~4 KB)
- ✅ `model_metadata.json` (~400 bytes)
- ✅ `selected_features.txt` (~2 KB)

These are already tracked by Git and will be deployed.

### Data Files for Prediction

Users will upload their own `.edf` or `.set` files through the Streamlit interface. You don't need to include sample data files in the repository.

## 🐛 Troubleshooting

### Issue: "File too large" error
**Solution**: Check `.gitignore` is properly excluding large files. Remove them from Git:
```powershell
git rm --cached processed_features.csv
git commit -m "Remove large data file"
```

### Issue: Deployment fails with dependency errors
**Solution**: Check `requirements.txt` has correct versions. Streamlit Cloud uses Python 3.11 by default.

### Issue: Model files not found
**Solution**: Ensure `.pkl` files are committed:
```powershell
git add *.pkl model_metadata.json selected_features.txt
git commit -m "Add model files"
git push
```

### Issue: App crashes on startup
**Solution**: Check Streamlit Cloud logs in the app dashboard for specific errors.

## 📊 Monitoring Your App

- **View logs**: Click "Manage app" → "Logs" in Streamlit Cloud
- **Reboot app**: Click "Reboot app" if it becomes unresponsive
- **Analytics**: View usage statistics in the Streamlit Cloud dashboard

## 🔒 Security Notes

- This app processes medical data - consider using **Private** repository if handling sensitive information
- Add authentication if needed (Streamlit supports password protection)
- Ensure compliance with HIPAA/GDPR if applicable

## 📞 Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Create issues in your repository

---

**Ready to deploy?** Follow the steps above and your app will be live in minutes! 🚀
