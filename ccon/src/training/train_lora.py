import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from src.utils.config import BASE_MODEL_NAME, LORA_OUTPUT_DIR, RAW_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger("train_lora")

def train_lora():
    logger.info("Starting LoRA training...")
    
    # Load dataset
    logger.info(f"Loading dataset from {RAW_DATA_PATH}")
    dataset = load_dataset("csv", data_files=str(RAW_DATA_PATH))
    
    # Preprocessing
    # In a real scenario, we would format this as an instruction tuning dataset
    # Input: "Rewrite this to [Target Culture]: [Input Text]"
    # Output: "[Rewritten Text]"
    
    model_name = "gpt2" # Using GPT2 as a lightweight placeholder for demo purposes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        inputs = [f"Rewrite to {culture}: {text}\nOutput: " for culture, text in zip(examples["target_culture"], examples["input_text"])]
        targets = [f"{rewrite}" for rewrite in examples["rewritten_text"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Load Model
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-3,
        logging_steps=10,
        save_steps=100,
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    logger.info("Training...")
    trainer.train()
    
    logger.info(f"Saving model to {LORA_OUTPUT_DIR}")
    model.save_pretrained(str(LORA_OUTPUT_DIR))

if __name__ == "__main__":
    train_lora()
