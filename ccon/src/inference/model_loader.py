import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.utils.config import BASE_MODEL_NAME, LORA_OUTPUT_DIR, CRSA_OUTPUT_DIR
from src.utils.logger import get_logger
import os

logger = get_logger("model_loader")

class ModelLoader:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.crsa_tokenizer = None
        self.crsa_model = None
        
    def load_models(self):
        logger.info("Loading models...")
        try:
            # Load Rewrite Model
            # For demo purposes, if LoRA weights don't exist, we might skip or load base only
            if os.path.exists(LORA_OUTPUT_DIR):
                logger.info(f"Loading LoRA from {LORA_OUTPUT_DIR}")
                base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
                self.model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
                self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                logger.warning("LoRA weights not found. Using mock/API mode.")

            # Load CRSA Model
            if os.path.exists(CRSA_OUTPUT_DIR):
                logger.info(f"Loading CRSA from {CRSA_OUTPUT_DIR}")
                self.crsa_tokenizer = AutoTokenizer.from_pretrained(str(CRSA_OUTPUT_DIR))
                self.crsa_model = AutoModelForSequenceClassification.from_pretrained(str(CRSA_OUTPUT_DIR))
            else:
                logger.warning("CRSA weights not found. Using mock mode.")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise e

model_loader = ModelLoader()
