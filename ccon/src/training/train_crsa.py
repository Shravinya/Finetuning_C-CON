import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from src.utils.config import CRSA_OUTPUT_DIR, RAW_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger("train_crsa")

def train_crsa():
    logger.info("Starting CRSA training...")
    
    # Mock data for risk classification since we don't have labeled risk data in the CSV yet
    # In a real scenario, we would load from a risk_labels.csv
    data = [
        {"text": "Fix this ASAP.", "label": 1}, # High Risk (Aggressive)
        {"text": "This is unacceptable.", "label": 1},
        {"text": "You are wrong.", "label": 1},
        {"text": "Could you please look into this?", "label": 0}, # Low Risk
        {"text": "Thank you for your help.", "label": 0},
        {"text": "I appreciate your effort.", "label": 0}
    ]
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
        
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=str(CRSA_OUTPUT_DIR),
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    logger.info("Training CRSA...")
    trainer.train()
    
    logger.info(f"Saving CRSA model to {CRSA_OUTPUT_DIR}")
    model.save_pretrained(str(CRSA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(CRSA_OUTPUT_DIR))

if __name__ == "__main__":
    train_crsa()
