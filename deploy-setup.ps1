# Quick Start Script for GitHub Deployment
# Run this script to initialize Git and prepare for deployment

Write-Host "🚀 Alzheimer's EEG Prediction - GitHub Deployment Setup" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✅ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed!" -ForegroundColor Red
    Write-Host "Please download and install Git from: https://git-scm.com/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if already a git repository
if (Test-Path ".git") {
    Write-Host "⚠️  Git repository already initialized" -ForegroundColor Yellow
    $response = Read-Host "Do you want to reinitialize? (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Skipping Git initialization..." -ForegroundColor Yellow
    } else {
        Remove-Item -Recurse -Force ".git"
        git init
        Write-Host "✅ Git repository reinitialized" -ForegroundColor Green
    }
} else {
    Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
}

Write-Host ""

# Check for required model files
Write-Host "🔍 Checking for required model files..." -ForegroundColor Yellow
$requiredFiles = @(
    "alzheimer_model.pkl",
    "scaler.pkl",
    "feature_selector.pkl",
    "model_metadata.json",
    "selected_features.txt",
    "app.py",
    "requirements.txt"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (MISSING!)" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️  WARNING: Missing required files!" -ForegroundColor Red
    Write-Host "Please ensure you have run the model training pipeline first." -ForegroundColor Yellow
    Write-Host "Missing files: $($missingFiles -join ', ')" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        exit 1
    }
}

Write-Host ""

# Add files to Git
Write-Host "📁 Adding files to Git..." -ForegroundColor Yellow
git add .
Write-Host "✅ Files added (respecting .gitignore)" -ForegroundColor Green

Write-Host ""

# Create initial commit
Write-Host "💾 Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Alzheimer's EEG Prediction App"
Write-Host "✅ Initial commit created" -ForegroundColor Green

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🎉 Git repository is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor White
Write-Host "   https://github.com/new" -ForegroundColor Blue
Write-Host ""
Write-Host "2. Name it: alzheimers-eeg-prediction (or your choice)" -ForegroundColor White
Write-Host ""
Write-Host "3. Run these commands (replace YOUR_USERNAME):" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/alzheimers-eeg-prediction.git" -ForegroundColor Yellow
Write-Host "   git branch -M main" -ForegroundColor Yellow
Write-Host "   git push -u origin main" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. Deploy to Streamlit Cloud:" -ForegroundColor White
Write-Host "   https://share.streamlit.io" -ForegroundColor Blue
Write-Host ""
Write-Host "📖 For detailed instructions, see DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Show repository status
Write-Host "📊 Repository Status:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Show file sizes
Write-Host "📦 Model File Sizes:" -ForegroundColor Cyan
Get-ChildItem -Filter "*.pkl" | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  $($_.Name): $size MB" -ForegroundColor White
}
Write-Host ""

Write-Host "✨ Setup complete! Ready to push to GitHub." -ForegroundColor Green
