#!/bin/bash
# Quick deployment script for Banana Leaf Disease Classifier

echo "=================================================="
echo "🍌 Banana Disease Classifier - Deployment Script"
echo "=================================================="
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "🔧 Initializing Git repository..."
    git init
    git branch -M main
fi

# Add all files
echo "📁 Adding files to Git..."
git add .

# Commit
echo "💾 Creating commit..."
read -p "Enter commit message (default: 'Banana disease classifier'): " commit_msg
commit_msg=${commit_msg:-"Banana disease classifier"}
git commit -m "$commit_msg"

# Add remote
echo ""
echo "🌐 Setting up GitHub remote..."
read -p "Enter your GitHub username: " github_user
read -p "Enter repository name (default: banana-disease-classifier): " repo_name
repo_name=${repo_name:-"banana-disease-classifier"}

remote_url="https://github.com/$github_user/$repo_name.git"
echo "Remote URL: $remote_url"

# Check if remote already exists
if git remote get-url origin &> /dev/null; then
    echo "Updating existing remote..."
    git remote set-url origin "$remote_url"
else
    echo "Adding new remote..."
    git remote add origin "$remote_url"
fi

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
git push -u origin main

echo ""
echo "=================================================="
echo "✅ Code pushed to GitHub!"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Go to https://github.com/$github_user/$repo_name"
echo "2. Go to https://render.com"
echo "3. Click 'New +' → 'Web Service'"
echo "4. Connect your GitHub repository"
echo "5. Set Build Command:"
echo "   pip install -r requirements.txt"
echo "6. Set Start Command:"
echo "   gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:\$PORT app:app"
echo "7. Add Environment Variables:"
echo "   - BACKUP_SVC = optional external verifier API key for fallback classification"
echo "8. Click Deploy"
echo ""
echo ""
echo "🎉 Your app will be live at: https://your-app-name.onrender.com"
echo ""
