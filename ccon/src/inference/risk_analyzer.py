import torch
from src.inference.model_loader import model_loader
from src.utils.logger import get_logger

logger = get_logger("risk_analyzer")

class RiskAnalyzer:
    def __init__(self):
        self.loader = model_loader
        
    def analyze_risk(self, text: str):
        """
        Analyzes the cultural risk of the input text.
        Returns a dictionary with risk level and details.
        """
        if self.loader.crsa_model:
            inputs = self.loader.crsa_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.loader.crsa_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            risk_score = probs[0][1].item() # Assuming label 1 is High Risk
            
            risk_level = "High" if risk_score > 0.5 else "Low"
            return {
                "risk_level": risk_level,
                "score": risk_score,
                "details": "Aggressive tone detected" if risk_level == "High" else "Tone appears neutral/polite"
            }
        else:
            # Mock logic for demo if model not loaded
            logger.info("Using mock risk analysis")
            keywords = ["asap", "fix", "wrong", "bad", "immediately", "unacceptable"]
            if any(k in text.lower() for k in keywords):
                return {
                    "risk_level": "High",
                    "score": 0.85,
                    "details": "Contains aggressive or demanding keywords."
                }
            return {
                "risk_level": "Low",
                "score": 0.1,
                "details": "No immediate cultural risks detected."
            }

risk_analyzer = RiskAnalyzer()
