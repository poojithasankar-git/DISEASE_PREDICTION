@echo off
REM Quick deployment script for Windows PowerShell

echo ===================================================
echo 🍌 Banana Disease Classifier - Deployment Setup
echo ===================================================
echo.

REM Check if git exists
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git is not installed. Please install Git first.
    exit /b 1
)

REM Initialize git
echo 🔧 Initializing Git repository...
git init
git branch -M main

REM Add files
echo 📁 Adding files to Git...
git add .

REM Commit
echo 💾 Creating commit...
git commit -m "Banana disease classifier - full stack application"

REM Get GitHub info
echo.
echo 🌐 GitHub Setup
echo ===================================================
set /p GITHUB_USER="Enter your GitHub username: "
set /p REPO_NAME="Enter repository name (default: banana-disease-classifier): "

if "%REPO_NAME%"=="" (
    set REPO_NAME=banana-disease-classifier
)

set REMOTE_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git
echo.
echo Remote URL: %REMOTE_URL%

REM Add remote
echo 🔗 Adding GitHub remote...
git remote add origin %REMOTE_URL% 2>nul || git remote set-url origin %REMOTE_URL%

REM Push
echo 🚀 Pushing to GitHub...
git push -u origin main

echo.
echo ===================================================
echo ✅ Code pushed to GitHub!
echo ===================================================
echo.
echo 📋 Next steps:
echo.
echo 1. Go to: https://github.com/%GITHUB_USER%/%REPO_NAME%
echo 2. Go to: https://render.com
echo 3. Click 'New +' -^> 'Web Service'
echo 4. Connect your GitHub repository
echo 5. Set Build Command:
echo    pip install -r requirements.txt
echo 6. Set Start Command:
echo    gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:$PORT app:app
echo 7. Add Environment Variables:
echo    - BACKUP_SVC = external verifier API key (optional, for fallback classification)
echo 8. Click Deploy
echo.
echo 🎉 Your app will be available at:
echo    https://your-app-name.onrender.com
echo.
pause
