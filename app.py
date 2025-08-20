import os
import json
import sqlite3
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from filelock import FileLock

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Path to the SQLite database (or JSON file as fallback)
HISTORY_DB = "history.db"
HISTORY_FILE = "history.json"

# Initialize SQLite database
def init_db():
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS history (text TEXT, prediction TEXT, confidence REAL)')

# Initialize the text classification model
text_classifier = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")

# Function to load history from SQLite (or JSON as fallback)
def load_history():
    try:
        with sqlite3.connect(HISTORY_DB) as conn:
            cursor = conn.execute('SELECT text, prediction, confidence FROM history')
            return [{"text": row[0], "prediction": row[1], "confidence": row[2]} for row in cursor]
    except sqlite3.Error:
        # Fallback to JSON if SQLite fails (e.g., initial deployment)
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        return []
                    return json.loads(content)
            return []
        except json.JSONDecodeError as e:
            print(f"Error loading history: {e}")
            return []

# Function to save history to SQLite (or JSON as fallback)
def save_history(history_entry):
    try:
        with sqlite3.connect(HISTORY_DB) as conn:
            conn.execute('INSERT INTO history (text, prediction, confidence) VALUES (?, ?, ?)',
                        (history_entry['text'], history_entry['prediction'], history_entry['confidence']))
    except sqlite3.Error:
        # Fallback to JSON
        with FileLock(HISTORY_FILE + ".lock"):
            history = load_history()
            history.append(history_entry)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=4)

# Initialize database at startup
init_db()

@app.route('/')
def index():
    """Render the homepage. History is not displayed here."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle text input for phishing detection."""
    try:
        if 'email_content' not in request.form:
            return jsonify({"error": "Missing email_content in form data"}), 400
        text_content = request.form['email_content']
        if not text_content.strip():
            return jsonify({"error": "Email content cannot be empty"}), 400

        result = text_classifier(text_content)
        label = result[0]['label']
        confidence = result[0]['score']

        if label == "benign":
            label = "Safe"
        elif label == "phishing":
            label = "Phishing"

        history_entry = {
            "text": text_content,
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        }

        save_history(history_entry)

        return jsonify({"label": label, "confidence": round(confidence * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Retrieve the history of predictions."""
    history = load_history()
    return jsonify(history)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
