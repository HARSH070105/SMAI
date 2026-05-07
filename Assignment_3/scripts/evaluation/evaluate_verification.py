import torch
import os
import sys
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, "scripts", "utils"))
from custom_prompts import custom_monument_prompts

ensemble_templates = [
    "A photo of the {}.",
    "The iconic {}.",
    "A historical landmark known as the {}.",
    "A famous Indian monument: the {}.",
    "A UNESCO World Heritage Site: the {}.",
    "An architectural photograph of the {}.",
    "The intricate stone carvings and architecture of the {}.",
    "The detailed exterior facade of the {}.",
    "The ornate structure of the {} in India.",
    "The ancient {} with its distinctive architectural style.",
    "A photo of the {} showing its stonework and design.",
    "The {} built from red sandstone.",
    "The {} built from white marble.",
    "The {} featuring domes and minarets.",
    "The {} featuring Dravidian temple architecture.",
    "A photo of the {} located in India.",
    "A tourist photo at the {}.",
    "Visitors at the {}.",
    "A photo of the {} against the sky.",
    "A photo of the {} from the ground looking up.",
    "A wide-angle shot of the {}.",
    "A close-up detail shot of the {}.",
    "A photo of the {} in bright daylight.",
    "A photo of the {} at golden hour.",
    "A nighttime photo of the {} illuminated by lights."
]

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP Models for Verification")
    parser.add_argument("--model_type", type=str, required=True, choices=["base", "baseline_lora", "ensemble_lora", "custom_ensemble_lora"])
    parser.add_argument("--prompt_type", type=str, required=True, choices=["baseline", "ensemble", "custom_ensemble"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"

    print(f"Loading {model_id} on {device}...")
    base_model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    if args.model_type == "base":
        model = base_model
    else:
        adapter_path = os.path.join(ROOT_DIR, "models", f"{args.model_type}_adapters")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path).to(device)

    model.eval()

    TEST_DIR = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "test")
    monuments_list = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])

    if not monuments_list:
        raise ValueError(f"No classes found in {TEST_DIR}")

    print(f"Generating text prototypes using '{args.prompt_type}' strategy...")
    prototype_vectors = []

    with torch.no_grad():
        for monument in tqdm(monuments_list, desc="Processing Prompts"):
            clean_name = monument.replace('_', ' ')
            
            if args.prompt_type == "baseline":
                sentences = [f"A photo of the {clean_name}."]
            elif args.prompt_type == "ensemble":
                sentences = [t.format(clean_name) for t in ensemble_templates]
            elif args.prompt_type == "custom_ensemble":
                matched_sentences = None
                if monument in custom_monument_prompts:
                    matched_sentences = custom_monument_prompts[monument]
                elif clean_name in custom_monument_prompts:
                    matched_sentences = custom_monument_prompts[clean_name]
                else:
                    for key, prompts in custom_monument_prompts.items():
                        if key.lower() in monument.lower() or monument.lower() in key.lower():
                            matched_sentences = prompts
                            break
                
                sentences = matched_sentences if matched_sentences else [f"A photo of the {clean_name}."]
            
            text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
            
            if hasattr(model, 'base_model'):
                text_outputs = model.base_model.get_text_features(**text_inputs)
            else:
                text_outputs = model.get_text_features(**text_inputs)
                
            if not isinstance(text_outputs, torch.Tensor):
                text_features = getattr(text_outputs, "text_embeds", getattr(text_outputs, "pooler_output", text_outputs[0]))
            else:
                text_features = text_outputs
                
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_vector = text_features.mean(dim=0)
            prototype_vector = mean_vector / mean_vector.norm(dim=-1, keepdim=True)
            prototype_vectors.append(prototype_vector)

    text_matrix = torch.stack(prototype_vectors).to(device)

    # Verification Data Arrays
    y_true_binary = [] # 1 for Genuine, 0 for Impostor
    y_scores = []      # The softmax confidence score

    print(f"\nStarting Verification Inference for Model: {args.model_type} | Prompt: {args.prompt_type}")
    with torch.no_grad():
        for true_class in monuments_list:
            class_folder = os.path.join(TEST_DIR, true_class)
            if not os.path.exists(class_folder):
                continue
                
            true_idx = monuments_list.index(true_class)
                
            for img_name in tqdm(os.listdir(class_folder), desc=f"Evaluating {true_class}"):
                img_path = os.path.join(class_folder, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")
                except:
                    continue
                    
                image_inputs = processor(images=image, return_tensors="pt").to(device)
                
                if hasattr(model, 'base_model'):
                    image_outputs = model.base_model.get_image_features(**image_inputs)
                else:
                    image_outputs = model.get_image_features(**image_inputs)
                    
                if not isinstance(image_outputs, torch.Tensor):
                    image_features = getattr(image_outputs, "image_embeds", getattr(image_outputs, "pooler_output", image_outputs[0]))
                else:
                    image_features = image_outputs
                    
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Softmax probabilities (Confidence scores from 0.0 to 1.0)
                similarity_scores = (100.0 * image_features @ text_matrix.T).softmax(dim=-1).cpu().numpy()[0]
                
                # Append the score for the TRUE class as a Genuine Match (1)
                y_true_binary.append(1)
                y_scores.append(similarity_scores[true_idx])
                
                # Append the scores for ALL OTHER classes as Impostor Matches (0)
                for c_idx in range(len(monuments_list)):
                    if c_idx != true_idx:
                        y_true_binary.append(0)
                        y_scores.append(similarity_scores[c_idx])

    # =====================================================================
    # Calculate Verification Metrics
    # =====================================================================
    y_true_binary = np.array(y_true_binary)
    y_scores = np.array(y_scores)

    # 1. ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    # 2. Equal Error Rate (EER) - The point where False Accept Rate == False Reject Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # 3. True Accept Rate at specific False Accept Rates (common in verification systems)
    tar_at_1_percent_far = interp1d(fpr, tpr)(0.01)
    tar_at_5_percent_far = interp1d(fpr, tpr)(0.05)

    print("\n" + "="*50)
    print(f"   VERIFICATION REPORT: {args.model_type.upper()} + {args.prompt_type.upper()}")
    print("="*50)
    print(f"ROC AUC Score:            {roc_auc:.4f}")
    print(f"Equal Error Rate (EER):   {eer*100:.2f}%")
    print(f"TAR @ 1.0% FAR:           {tar_at_1_percent_far*100:.2f}%")
    print(f"TAR @ 5.0% FAR:           {tar_at_5_percent_far*100:.2f}%")

    # Save to dynamic report folder
    report_dir = os.path.join(ROOT_DIR, "reports", "verification", args.model_type, f"{args.prompt_type}_prompts")
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "verification_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"   VERIFICATION REPORT: {args.model_type.upper()} + {args.prompt_type.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"ROC AUC Score:            {roc_auc:.4f}\n")
        f.write(f"Equal Error Rate (EER):   {eer*100:.2f}%\n")
        f.write(f"TAR @ 1.0% FAR:           {tar_at_1_percent_far*100:.2f}%\n")
        f.write(f"TAR @ 5.0% FAR:           {tar_at_5_percent_far*100:.2f}%\n")

    # =====================================================================
    # Plotting: Genuine vs. Impostor Distributions
    # =====================================================================
    genuine_scores = y_scores[y_true_binary == 1]
    impostor_scores = y_scores[y_true_binary == 0]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Probability Density Histograms
    sns.kdeplot(impostor_scores, fill=True, color="red", label="Impostor Matches (Wrong Photos)", ax=axes[0])
    sns.kdeplot(genuine_scores, fill=True, color="green", label="Genuine Matches (Correct Photos)", ax=axes[0])
    axes[0].set_title("Distribution of Confidence Scores", fontsize=14)
    axes[0].set_xlabel("Model Confidence Score (Softmax Probability)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot 2: ROC Curve
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # Mark EER point
    axes[1].plot(eer, 1-eer, marker='o', markersize=8, color="red", label=f"EER ({eer*100:.2f}%)")
    
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate (Accidental Accepts)", fontsize=12)
    axes[1].set_ylabel("True Positive Rate (Correct Accepts)", fontsize=12)
    axes[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(report_dir, "verification_plots.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plots to: {report_dir}")

if __name__ == "__main__":
    main()