# Banana Leaf Disease Classifier - Full Stack

A full stack app with a Flask API and a web UI for banana leaf disease detection.

## Stack

- Flask API
- TensorFlow/Keras model
- HTML/CSS/JS frontend
- Render deployment

## Deploy (Render)

Build Command:

pip install -r requirements.txt

Start Command:

gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:$PORT app:app

Optional env var:

BACKUP_SVC = external verifier API key

## Project Structure

- app.py
- requirements.txt
- render.yaml
- Procfile
- templates/
- static/
- models/banana_model.h5
- models/metadata.json

## Local Run

pip install -r requirements.txt
python app.py
