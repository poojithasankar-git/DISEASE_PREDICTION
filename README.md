# Banana Leaf Disease Classifier

A Flask API that serves a pre-trained ResNet50 model for banana leaf disease detection.

## Features

- Inference-only deployment (no training on deploy)
- Green pixel validation to reject non-leaf images
- Confidence gating for predictions
- Optional external fallback verification (hidden) via BACKUP_SVC

## Quick Start (Render)

Build Command:

pip install -r requirements.txt

Start Command:

gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:$PORT app:app

Optional env var:

BACKUP_SVC = external verifier API key

## API Endpoints

- GET /health
- GET /info
- POST /predict

## Model Files

- models/banana_model.h5
- models/metadata.json

## Local Run

pip install -r requirements.txt
python app.py
