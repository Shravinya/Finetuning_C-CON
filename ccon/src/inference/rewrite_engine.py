import os
from groq import Groq
from src.inference.model_loader import model_loader
from src.inference.style_blender import style_blender
from src.utils.logger import get_logger
from src.utils.config import GROQ_API_KEY

logger = get_logger("rewrite_engine")

# Initialize Groq client
client = Groq(
    api_key=GROQ_API_KEY,
)

class RewriteEngine:
    def __init__(self):
        self.loader = model_loader
        self.blender = style_blender

    def rewrite(self, text: str, target_culture: str, blend_culture: str = None, blend_weight: int = 50):
        logger.info(f"Rewriting text: '{text}' for {target_culture}")
        
        if blend_culture:
            instruction = self.blender.blend_styles(target_culture, blend_culture, blend_weight)
            prompt = f"{instruction}\nOriginal Text: {text}\nRewritten Text:"
        else:
            prompt = f"Rewrite the following text to match {target_culture} cultural norms and communication style.\nOriginal Text: {text}\nRewritten Text:"

        # Try using local model if loaded
        if self.loader.model:
            # Local inference logic would go here
            # For now, we prioritize Groq as requested for "amazing" results, 
            # but we could fallback to local if Groq fails or if user explicitly wants local.
            pass
        
        # Use Groq for high-quality inference
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are C-CON, a cultural context rewriter. Rewrite texts to fit specific cultural norms perfectly."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API failed: {e}")
            return f"Error: Could not rewrite text. {e}"

rewrite_engine = RewriteEngine()
