# 📘 C-CON Technical Implementation Report

This document provides a comprehensive technical overview of the **C-CON (Cultural Context Rewriter)** project, detailing the architecture, data pipeline, model training, and inference strategies implemented from scratch.

---

## 1. 🏗️ System Architecture
We designed a modular, industry-standard directory structure to ensure scalability and maintainability:

- **`data/`**: Centralized storage for raw (`.csv`) and processed datasets.
- **`src/`**: Source code split into logical modules:
  - **`training/`**: Scripts for model fine-tuning.
  - **`inference/`**: Core logic for text rewriting and analysis.
  - **`api/`**: FastAPI backend for external access.
  - **`utils/`**: Shared helper functions (logging, config).
- **`web/`**: Streamlit-based frontend for user interaction.

This structure separates **training concerns** (data science) from **application concerns** (software engineering).

---

## 2. 📊 Data Engineering
### Synthetic Dataset Generation
Since no public dataset existed for this specific task, we created a **synthetic dataset** (`data/raw/ccon_dataset.csv`) containing pairs of:
- **Input Text**: (e.g., "Fix this ASAP")
- **Target Culture**: (e.g., "Japanese Polite", "Indian Corporate")
- **Rewritten Text**: (e.g., "I apologize for the urgency, but could you please address this...")

**Key Action**: We expanded this dataset to **50+ examples** covering diverse scenarios (deadlines, feedback, disagreements) to improve model generalization.

---

## 3. 🧠 Model Fine-Tuning (The "AI" Core)
We implemented two distinct training pipelines:

### A. LoRA Fine-Tuning (`train_lora.py`)
- **Goal**: Teach a Large Language Model (LLM) to rewrite text in specific cultural styles.
- **Technique**: **LoRA (Low-Rank Adaptation)**. Instead of retraining all model weights (which is expensive), we freeze the base model and train only small adapter layers.
- **Implementation**:
  - Used `peft` library for LoRA configuration.
  - Used `transformers` `Trainer` API.
  - **Base Model**: Configured to use `gpt2` for local demonstration (CPU-friendly), but the code is ready for `Llama-3` or `Mistral`.

### B. Cultural Risk & Sensitivity Analyzer (CRSA) (`train_crsa.py`)
- **Goal**: Detect if a message carries "High Risk" (aggressive, rude) or "Low Risk" (polite).
- **Technique**: Sequence Classification.
- **Implementation**:
  - Fine-tuned a **DistilBERT** model.
  - Binary classification (Risk vs. No Risk).

---

## 4. 🚀 Inference Engine & Groq Integration
To achieve "amazing" real-time results, we evolved the inference strategy:

1.  **Initial Approach**: OpenAI API (hit quota limits).
2.  **Fallback Approach**: Local Model (good for privacy, but resource-heavy).
3.  **Final "Pro" Approach**: **Groq API Integration**.
    - We integrated **Groq** to access **`llama-3.3-70b-versatile`**.
    - **Why?** It offers state-of-the-art reasoning and speed, ensuring the rewrites are nuanced and culturally accurate.

### Key Components:
- **`RewriteEngine`**: Orchestrates the logic. It constructs prompts that instruct the LLM to adopt specific cultural personas.
- **`StyleBlender`**: Implements **Adaptive Cultural Style Blending (ACSB)**. It mathematically weights instructions (e.g., "70% Japanese, 30% American") to generate hybrid communication styles.

---

## 5. 💻 Application Layer
### Backend (FastAPI)
- Created a robust API (`src/api/app.py`) with endpoints:
  - `/rewrite`: Handles the core logic.
  - `/analyze_risk`: Exposes the CRSA model.
- Used **Pydantic** schemas for strict data validation.

### Frontend (Streamlit)
- Built a modern, interactive UI (`web/streamlit_app.py`).
- Features:
  - Real-time text input.
  - Dynamic culture selection.
  - **Risk Dashboard**: Visual feedback (Green/Red) based on CRSA analysis.
  - **Blending Sliders**: Interactive controls for style mixing.

---

## 6. 🛠️ Challenges & Solutions
| Challenge | Solution |
|-----------|----------|
| **OpenAI Quota Limits** | Switched to **Groq API** for free, high-speed, high-quality inference. |
| **Import Errors** | Fixed Python path issues by dynamically appending the project root to `sys.path` in the app scripts. |
| **Lack of Data** | Manually engineered a rich synthetic dataset with diverse cultural examples. |

---

## ✅ Summary
We have built an **end-to-end AI system** that goes beyond simple translation. It understands **nuance, tone, and culture**. By combining **LoRA fine-tuning code**, **risk classification**, and **state-of-the-art LLM inference**, C-CON is a production-ready prototype.
