from transformers import pipeline

# Load the fine-tuned model
pipe = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")
