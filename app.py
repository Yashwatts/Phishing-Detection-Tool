import os
import json
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from filelock import FileLock

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Path to the JSON file that stores history
HISTORY_FILE = "history.json"

# Initialize the text classification model
text_classifier = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")

# Function to load history from the JSON file
def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                content = f.read().strip()
                if not content:  # Handle empty file
                    return []
                return json.loads(content)
        return []  # Return empty list if file doesn't exist
    except json.JSONDecodeError as e:
        print(f"Error loading history: {e}")
        return []  # Return empty list on error

# Function to save history to the JSON file
def save_history(history):
    with FileLock(HISTORY_FILE + ".lock"):
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)

@app.route('/')
def index():
    """Render the homepage. History is not displayed here."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle text input for phishing detection."""
    try:
        # Get the email content from the form
        if 'email_content' not in request.form:
            return jsonify({"error": "Missing email_content in form data"}), 400
        text_content = request.form['email_content']
        if not text_content.strip():
            return jsonify({"error": "Email content cannot be empty"}), 400

        # Perform prediction using the text classifier
        result = text_classifier(text_content)
        label = result[0]['label']
        confidence = result[0]['score']

        # Map labels to user-friendly values
        if label == "benign":
            label = "Safe"
        elif label == "phishing":
            label = "Phishing"

        # Create a history entry
        history_entry = {
            "text": text_content,
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        }

        # Load and update history
        history = load_history()
        history.append(history_entry)
        save_history(history)

        # Return the result to the user
        return jsonify({"label": label, "confidence": round(confidence * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Retrieve the history of predictions from the JSON file. This endpoint is for localhost inspection."""
    history = load_history()
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)