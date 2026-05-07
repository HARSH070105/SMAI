# Monument Scavenger Hunt Game

**Team Name:** Stress, Math and A.I.

## The Problem

Tourists often want to know more about the historical monuments they are visiting, but traditional searches take them out of the moment. Furthermore, exploring heritage sites can sometimes lack an engaging, interactive element.

## What We Built

This project is a gamified Streamlit web application that transforms standard monument identification into a scavenger hunt.

- The app generates cryptic clues about a specific Indian monument.
- The user figures out the clue and uploads a photograph of the correct monument.
- Using a Finetuned Zero-Shot AI vision model (CLIP + LoRA Adapters), the app verifies if the user found the right place.
- Upon success, it rewards the user with a historical paragraph, a visit info card (opening hours and ticket prices), and an "Open in Google Maps" link.

## Features

- **Gamified Experience:** Users are challenged with clues rather than just identifying random images.
- **Robust Image Classification & Verification:** Utilizes OpenAI's `clip-vit-base-patch32` with customized custom prompt engineering and LoRA parameter-efficient fine-tuning (PEFT).
- **Rich Metadata Integration:** Displays cached JSON data (history, hours, ticket prices) scraped from Wikipedia/Gemini.
- **Seamless Navigation:** Direct integration with Google Maps to help users locate the monument.

## Model Evaluation and Results 

We fine-tuned the base CLIP model using LoRA adapters across multiple prompting strategies (Baseline, Ensemble, and Custom Ensemble).

### Classification Accuracy

Our models achieved a significant performance boost during zero-shot classification evaluation through Fine-Tuning:

- **Base CLIP (Custom Ensemble):** ~66.80%
- **Custom Ensemble LoRA (Custom Ensemble Prompts):** **92.34%** (Best overall)
- **Baseline LoRA (Baseline Prompts):** 90.27%

### Verification Metrics

For genuine monument verification matching, we dramatically reduced the Equal Error Rate (EER) and boosted the True Accept Rate (TAR) at 1% False Accept Rate (FAR):

- **EER (Lower is better):** Reduced from ~6.37% (Base) down to **1.67%** (Ensemble LoRA with baseline prompts) and **1.79%** (Custom Ensemble LoRA).
- **TAR @ 1.0% FAR (Higher is better):** Increased from ~72.67% (Base) up to **97.38%** (Custom Ensemble LoRA with customized prompts).
- **ROC AUC:** Up to **0.9990** post-finetuning.

## Project Structure 

```text
SMAI_A3/
├── data/ & metadata/      # JSON details scraped for 24+ Indian monuments
├── models/                # LoRA Adapters (baseline_lora, custom_ensemble_lora, ensemble_lora)
├── reports/               # In-depth classification and verification matrices & Markdown reports
├── scripts/
│   ├── app/               # Streamlit application source code (`app.py`, `dataset_utils.py`)
│   ├── data_collection/   # Scrapers, class expansion, data preparation scripts
│   ├── evaluation/        # Validation & Metrics building (`evaluate_classification.py`, etc)
│   ├── finetuning/        # PyTorch & HuggingFace training loops using PEFT/LoRA
│   └── inference/         # Model execution wrappers (`verify_monument.py`)
└── README.md
```

## Tech Stack & Skills

- **Frontend:** Streamlit
- **Machine Learning:** Zero-shot classification & Verification using `openai/clip-vit-base-patch32` via Hugging Face Transformers.
- **Training:** Parameter-Efficient Finetuning (PEFT/LoRA) for CLIP.
- **Data Sources:** Indian Monuments Image Dataset (Danushkumarv, 24 monuments, ~3.5k images) & Wikimedia Commons.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HARSH070105/SMAI_A3.git
   cd SMAI_A3
   ```

2. **Install the dependencies:**
   Make sure you have a Python environment set up (e.g., venv or conda).
   ```bash
   pip install -r scripts/app/requirements.txt
   ```

3. **Start the Streamlit app:**
   ```bash
   streamlit run scripts/app/app.py
   ```
   *The app should automatically open in your default browser at `http://localhost:8501`.*
