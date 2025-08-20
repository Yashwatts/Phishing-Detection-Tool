from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the phishing dataset (train.csv and test.csv)
dataset = load_dataset('csv', data_files={'train': 'data/train.csv', 'test': 'data/test.csv'})

# Load the pre-trained model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments (configurations for the training process)
training_args = TrainingArguments(
    output_dir="./results",               # Directory to save the model
    evaluation_strategy="epoch",          # Evaluate after every epoch
    learning_rate=2e-5,                   # Learning rate
    per_device_train_batch_size=16,       # Batch size during training
    num_train_epochs=3,                   # Number of epochs
    weight_decay=0.01,                    # Weight decay for optimization
    logging_dir='./logs',                 # Directory for logs
)

# Initialize Trainer with the model, arguments, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],  # Evaluate using the test dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./phishing-model")
tokenizer.save_pretrained("./phishing-model")
