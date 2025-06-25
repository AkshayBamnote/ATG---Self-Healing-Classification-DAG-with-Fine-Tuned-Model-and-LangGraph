from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os
import transformers

print(f"Transformers version: {transformers.__version__}")

# Create necessary folders
os.makedirs("models/sentiment_model", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

encoded = dataset.map(preprocess, batched=True)
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained model for classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
args = TrainingArguments(
    output_dir="models/sentiment_model",
    per_device_train_batch_size=16,
    num_train_epochs=1,
    eval_strategy="epoch",  
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=1
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=encoded["test"].select(range(500)),
)

trainer.train()

# Save model and tokenizer
model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")

print(" Model training complete and saved to models/sentiment_model")
