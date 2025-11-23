import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_PATH = DATA_DIR / "raw" / "ccon_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

# Model Configs
BASE_MODEL_NAME = "gpt2" # Using GPT2 for CPU-friendly demo
LORA_OUTPUT_DIR = MODELS_DIR / "lora"
CRSA_OUTPUT_DIR = MODELS_DIR / "crsa"

# App Configs
API_HOST = "0.0.0.0"
API_PORT = 8000
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
