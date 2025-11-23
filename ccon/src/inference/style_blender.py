from src.utils.logger import get_logger

logger = get_logger("style_blender")

class StyleBlender:
    def blend_styles(self, culture_a: str, culture_b: str, weight_a: float):
        """
        Returns a prompt instruction for blending styles.
        """
        weight_b = 100 - weight_a
        instruction = (
            f"Rewrite the text by blending {weight_a}% {culture_a} style and {weight_b}% {culture_b} style. "
            f"Maintain the core message but balance the tone accordingly."
        )
        return instruction

style_blender = StyleBlender()
