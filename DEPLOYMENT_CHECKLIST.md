# 📋 GitHub Deployment Checklist

Use this checklist to ensure everything is ready for deployment.

## ✅ Pre-Deployment Checklist

### Required Files
- [ ] `app.py` - Main Streamlit application
- [ ] `requirements.txt` - Python dependencies
- [ ] `.gitignore` - Excludes large files
- [ ] `README.md` - Project documentation
- [ ] `LICENSE` - MIT License
- [ ] `DEPLOYMENT.md` - Deployment guide
- [ ] `.streamlit/config.toml` - Streamlit configuration
- [ ] `packages.txt` - System dependencies

### Model Files (Must be present!)
- [ ] `alzheimer_model.pkl` (~2.3 MB)
- [ ] `scaler.pkl` (~23 KB)
- [ ] `feature_selector.pkl` (~4 KB)
- [ ] `model_metadata.json` (~400 bytes)
- [ ] `selected_features.txt` (~2 KB)

### Optional but Recommended
- [ ] `CONTRIBUTING.md` - Contribution guidelines
- [ ] `.github/workflows/ci.yml` - GitHub Actions CI
- [ ] `setup.sh` - Streamlit Cloud setup script
- [ ] `deploy-setup.ps1` - Deployment automation script

## 🚀 Deployment Steps

### Step 1: Initialize Git
- [ ] Run `.\deploy-setup.ps1` OR manually:
  - [ ] `git init`
  - [ ] `git add .`
  - [ ] `git commit -m "Initial commit: Alzheimer's EEG Prediction App"`

### Step 2: Create GitHub Repository
- [ ] Go to https://github.com/new
- [ ] Repository name: `alzheimers-eeg-prediction` (or your choice)
- [ ] Description: "ML-powered Alzheimer's stage prediction from EEG data"
- [ ] Choose Public or Private
- [ ] **DO NOT** initialize with README, .gitignore, or license
- [ ] Click "Create repository"

### Step 3: Push to GitHub
- [ ] `git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git`
- [ ] `git branch -M main`
- [ ] `git push -u origin main`
- [ ] Verify all files uploaded successfully on GitHub

### Step 4: Deploy to Streamlit Cloud
- [ ] Go to https://share.streamlit.io
- [ ] Sign in with GitHub account
- [ ] Click "New app"
- [ ] Select your repository
- [ ] Branch: `main`
- [ ] Main file: `app.py`
- [ ] Choose custom URL (e.g., `alzheimers-eeg-predictor`)
- [ ] Click "Deploy"
- [ ] Wait for deployment (2-5 minutes)

### Step 5: Test Deployment
- [ ] App loads without errors
- [ ] Can upload EDF files
- [ ] Predictions work correctly
- [ ] LIME explanations display
- [ ] Model metrics show correctly
- [ ] No console errors

### Step 6: Update README
- [ ] Replace `https://your-app-name.streamlit.app` with actual URL
- [ ] Update badge link in README.md
- [ ] Commit and push changes

## 🔍 Verification

### File Size Check
Run this to check model file sizes:
```powershell
Get-ChildItem -Filter "*.pkl" | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host "$($_.Name): $size MB"
}
```

All `.pkl` files should be under 100 MB (GitHub limit).

### Git Status Check
```bash
git status
```
Should show clean working tree after initial commit.

### Test Locally First
```bash
streamlit run app.py
```
Ensure app works locally before deploying.

## ⚠️ Common Issues

### Issue: "File too large" error
- [ ] Check `.gitignore` excludes large data files
- [ ] Remove tracked large files: `git rm --cached filename`

### Issue: Missing model files
- [ ] Ensure all `.pkl` files are committed
- [ ] Check they're not in `.gitignore`
- [ ] Force add if needed: `git add -f *.pkl`

### Issue: Deployment fails
- [ ] Check Streamlit Cloud logs
- [ ] Verify `requirements.txt` versions
- [ ] Ensure Python 3.11 compatibility

### Issue: App crashes on startup
- [ ] Check all model files are in repository
- [ ] Verify file paths in `app.py`
- [ ] Check Streamlit Cloud logs for errors

## 📊 Post-Deployment

### Monitoring
- [ ] Check app analytics in Streamlit Cloud dashboard
- [ ] Monitor error logs
- [ ] Test with different EEG files

### Sharing
- [ ] Share app URL with team/users
- [ ] Add to portfolio/resume
- [ ] Share on social media (optional)

### Maintenance
- [ ] Set up notifications for app downtime
- [ ] Plan regular updates
- [ ] Monitor user feedback

## 🎉 Success Criteria

Your deployment is successful when:
- ✅ App is accessible via public URL
- ✅ Users can upload EEG files
- ✅ Predictions are accurate
- ✅ No errors in logs
- ✅ App is responsive and fast

---

**Need help?** See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Questions?** Open an issue on GitHub or check Streamlit Community Forum.
