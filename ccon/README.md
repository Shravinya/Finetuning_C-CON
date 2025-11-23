# C-CON: Cultural Context Rewriter

C-CON is an AI model that rewrites text into a communication style that fits the cultural expectations of different countries.

## Features
- **Cultural Rewriting**: Rewrite text to match specific cultural norms (e.g., Japanese Polite, American Direct).
- **Cultural Risk & Sensitivity Analyzer (CRSA)**: Analyze text for cultural risks before sending.
- **Adaptive Cultural Style Blending (ACSB)**: Blend multiple cultural styles.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API:
   ```bash
   uvicorn src.api.app:app --reload
   ```

3. Run the UI:
   ```bash
   streamlit run web/streamlit_app.py
   ```
