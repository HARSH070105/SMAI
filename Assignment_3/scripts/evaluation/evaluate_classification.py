import torch
import os
import sys
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    parser = argparse.ArgumentParser(description="Evaluate CLIP Models")
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
        model_name = "Base CLIP"
    else:
        adapter_path = os.path.join(ROOT_DIR, "models", f"{args.model_type}_adapters")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        model_name = args.model_type

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
                
                if matched_sentences:
                    sentences = matched_sentences
                else:
                    print(f"Warning: No custom prompt found for '{monument}', falling back to baseline.")
                    sentences = [f"A photo of the {clean_name}."]
            
            text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
            
            # Text encoder is always evaluated using base_model
            # Note: if the model is PeftModel, get_text_features might need base_model
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

    y_true, y_pred = [], []

    print(f"\nStarting inference for Model: {args.model_type} | Prompt: {args.prompt_type}")
    with torch.no_grad():
        for true_class in monuments_list:
            class_folder = os.path.join(TEST_DIR, true_class)
            if not os.path.exists(class_folder):
                continue
                
            for img_name in tqdm(os.listdir(class_folder), desc=f"Evaluating {true_class}"):
                img_path = os.path.join(class_folder, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")
                except:
                    continue
                    
                image_inputs = processor(images=image, return_tensors="pt").to(device)
                
                # Image encoder uses the peft model (if applicable)
                if hasattr(model, 'base_model'):
                    image_outputs = model.base_model.get_image_features(**image_inputs)
                else:
                    image_outputs = model.get_image_features(**image_inputs)
                    
                if not isinstance(image_outputs, torch.Tensor):
                    image_features = getattr(image_outputs, "image_embeds", getattr(image_outputs, "pooler_output", image_outputs[0]))
                else:
                    image_features = image_outputs
                    
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarity_scores = (100.0 * image_features @ text_matrix.T).softmax(dim=-1)
                predicted_idx = similarity_scores.argmax().item()
                predicted_class = monuments_list[predicted_idx]
                
                y_true.append(true_class)
                y_pred.append(predicted_class)

    accuracy_str = f"Overall Top-1 Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%"
    report_str = classification_report(y_true, y_pred)

    print("\n" + "="*50)
    print(f"   {args.model_type.upper()} + {args.prompt_type.upper()} PROMPTS")
    print("="*50)
    print(accuracy_str)

    # Save to dynamic report folder
    report_dir = os.path.join(ROOT_DIR, "reports", "classification", args.model_type, f"{args.prompt_type}_prompts")
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"   {args.model_type.upper()} + {args.prompt_type.upper()} PROMPTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(accuracy_str + "\n\n")
        f.write("Detailed Report:\n")
        f.write(report_str + "\n")

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = os.path.join(report_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)

    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, vmin=0.0, vmax=1.0, ax=ax,
        annot_kws={"size": 6}
    )
    ax.set_title(f"Normalized Confusion Matrix\nModel: {args.model_type} | Prompt: {args.prompt_type}\n{accuracy_str}", fontsize=18, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved report to: {report_dir}")

if __name__ == "__main__":
    main()
