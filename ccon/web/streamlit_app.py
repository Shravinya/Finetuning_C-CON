import streamlit as st
import requests
import json
import os
import sys

# Add project root to path so src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="C-CON: Cultural Context Rewriter",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box_shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 C-CON: Cultural Context Rewriter")
st.markdown("### Transform your communication to fit any culture instantly.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Message")
    input_text = st.text_area("Enter your message here...", height=150, placeholder="e.g., Fix this ASAP.")
    
    st.subheader("🎯 Target Culture")
    target_culture = st.selectbox(
        "Select Target Culture",
        ["American Direct", "Japanese Polite", "Indian Corporate", "German Direct", "Middle Eastern Respectful", "British Indirect"]
    )
    
    st.subheader("🔀 Style Blending (Optional)")
    enable_blending = st.checkbox("Enable Cultural Blending")
    blend_culture = None
    blend_weight = 50
    
    if enable_blending:
        blend_culture = st.selectbox(
            "Select Second Culture to Blend",
            ["American Direct", "Japanese Polite", "Indian Corporate", "German Direct", "Middle Eastern Respectful"]
        )
        blend_weight = st.slider(f"Weight for {target_culture} (%)", 0, 100, 70)

    if st.button("Rewrite Message"):
        if not input_text:
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Analyzing cultural nuances and rewriting..."):
                try:
                    # Call API (Assuming it's running, otherwise use direct logic for demo if API fails)
                    payload = {
                        "text": input_text,
                        "target_culture": target_culture,
                        "blend_culture": blend_culture if enable_blending else None,
                        "blend_weight": blend_weight
                    }
                    
                    # Direct import for demo if API is not running separately
                    # This is a hack for the single-process demo environment
                    from src.inference.rewrite_engine import rewrite_engine
                    from src.inference.risk_analyzer import risk_analyzer
                    
                    risk = risk_analyzer.analyze_risk(input_text)
                    rewritten = rewrite_engine.rewrite(
                        input_text, 
                        target_culture, 
                        blend_culture if enable_blending else None, 
                        blend_weight
                    )
                    
                    st.session_state.result = {
                        "rewritten": rewritten,
                        "risk": risk
                    }
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with col2:
    st.subheader("✨ Results")
    
    if "result" in st.session_state:
        res = st.session_state.result
        
        # Risk Report
        risk = res["risk"]
        risk_color = "red" if risk["risk_level"] == "High" else "green"
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 8px; background-color: {'#ffebee' if risk['risk_level'] == 'High' else '#e8f5e9'}; border: 1px solid {risk_color}; margin-bottom: 1rem;">
            <h4 style="color: {risk_color}; margin: 0;">🛡️ Cultural Risk Analysis: {risk['risk_level']}</h4>
            <p style="margin: 0.5rem 0 0 0;">{risk['details']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Rewritten Text
        st.markdown("#### ✍️ Rewritten Message")
        st.info(res["rewritten"])
        
        st.markdown("---")
        st.button("Copy to Clipboard", help="Copy the rewritten text")

st.markdown("---")
st.markdown("Built with ❤️ by C-CON Team | Powered by LoRA & LLMs")
