from fastapi import FastAPI, HTTPException
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.schemas import RewriteRequest, RewriteResponse, RiskAnalysisRequest, RiskAnalysisResponse
from src.inference.rewrite_engine import rewrite_engine
from src.inference.risk_analyzer import risk_analyzer
from src.utils.logger import get_logger

logger = get_logger("api")
app = FastAPI(title="C-CON API", description="Cultural Context Rewriter API")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "C-CON API is running"}

@app.post("/rewrite", response_model=RewriteResponse)
def rewrite_text(request: RewriteRequest):
    logger.info(f"Received rewrite request: {request}")
    try:
        # Step 1: Analyze Risk (Optional, but good to include)
        risk = risk_analyzer.analyze_risk(request.text)
        
        # Step 2: Rewrite
        rewritten = rewrite_engine.rewrite(
            request.text, 
            request.target_culture, 
            request.blend_culture, 
            request.blend_weight
        )
        
        return RewriteResponse(
            original_text=request.text,
            rewritten_text=rewritten,
            target_culture=request.target_culture,
            risk_analysis=risk
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_risk", response_model=RiskAnalysisResponse)
def analyze_risk_endpoint(request: RiskAnalysisRequest):
    try:
        result = risk_analyzer.analyze_risk(request.text)
        return RiskAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
