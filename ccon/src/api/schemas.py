from pydantic import BaseModel
from typing import Optional

class RewriteRequest(BaseModel):
    text: str
    target_culture: str
    blend_culture: Optional[str] = None
    blend_weight: Optional[int] = 50

class RewriteResponse(BaseModel):
    original_text: str
    rewritten_text: str
    target_culture: str
    risk_analysis: Optional[dict] = None

class RiskAnalysisRequest(BaseModel):
    text: str

class RiskAnalysisResponse(BaseModel):
    risk_level: str
    score: float
    details: str
