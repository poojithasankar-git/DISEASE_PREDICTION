# Deployment Guide - Banana Leaf Disease Classifier

## Overview

This is a pre-trained, inference-only deployment. No training runs on Render.

- Model: models/banana_model.h5
- Metadata: models/metadata.json
- Optional fallback: external verifier via BACKUP_SVC

## Quick Deployment (Render)

1. Create a new Web Service in Render.
2. Set Build Command:
   pip install -r requirements.txt
3. Set Start Command:
   gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:$PORT app:app
4. (Optional) Add env var:
   BACKUP_SVC = your external verifier API key
5. Deploy.

## Local Run

1. Install dependencies:
   pip install -r requirements.txt
2. Run server:
   python app.py
3. Test:
   curl http://localhost:5000/health

## Updating the Model

To ship a new model:

1. Replace models/banana_model.h5 and models/metadata.json
2. Commit and push
3. Redeploy on Render

## Troubleshooting

- Model not found: ensure models/banana_model.h5 is tracked in git.
- Slow start: first request warms the model; retry after 30-60 seconds.
- Low confidence: try a clearer banana leaf image.
