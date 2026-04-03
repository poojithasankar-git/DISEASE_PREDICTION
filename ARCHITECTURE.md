# Banana Leaf Disease Classifier - Architecture

## Overview

This project is inference-only. A pre-trained ResNet50 model is loaded into memory and used for predictions.

## Components

- Frontend: templates/index.html, static/style.css, static/script.js
- Backend: app.py (Flask API)
- Model: models/banana_model.h5
- Metadata: models/metadata.json

## Data Flow

1. User uploads an image in the web UI
2. Flask validates the file and checks green pixel ratio
3. Image is resized to 224x224 and preprocessed
4. ResNet50 model predicts class probabilities
5. Confidence gate and optional external verification
6. JSON response returned to UI

## Deployment

- Build Command: pip install -r requirements.txt
- Start Command: gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:$PORT app:app
- Optional env var: BACKUP_SVC (external verifier API key)

## Notes

- Model is loaded lazily on first request
- No Kaggle credentials needed
- No training runs on deploy
