import os
import argparse
import sys
from PIL import Image

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Add utils and baseline inference to path
sys.path.append(os.path.join(ROOT_DIR, "scripts", "utils"))
sys.path.append(os.path.join(ROOT_DIR, "scripts", "inference", "baseline"))

from custom_prompts import custom_monument_prompts
from inference import load_model, precompute_text_features, run_inference

def verify_monument(image_path, target_class, threshold=0.5, model_type="custom_ensemble_lora", prompt_type="custom_ensemble"):
    """
    Verifies if an image belongs to the target class based on a probability threshold.
    """
    # Load all classes to form a proper softmax distribution
    all_classes = list(custom_monument_prompts.keys())
    
    # If the user provides a class name not in our dictionary, add it to the classification pool
    if target_class not in all_classes:
        all_classes.append(target_class)

    # Resolve adapter path
    if model_type == "base":
        lora_weights_path = None
    else:
        lora_weights_path = os.path.join(ROOT_DIR, "models", f"{model_type}_adapters")
        if not os.path.exists(lora_weights_path):
            print(f"Warning: Adapter not found at {lora_weights_path}. Falling back to base model.")
            lora_weights_path = None

    model, processor, device = load_model(lora_weights_path=lora_weights_path)

    # Determine whether to use custom ensemble text prototypes
    use_custom = True if prompt_type in ["ensemble", "custom_ensemble"] else False
    
    print("\nPrecomputing text embeddings for verification pool...")
    text_features = precompute_text_features(model, processor, all_classes, device, use_custom_ensemble=use_custom)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return False

    print("Running inference...")
    predicted_class, probabilities = run_inference(image, model, processor, text_features, all_classes, device)

    # Check the probability of the target class
    target_prob = probabilities.get(target_class, 0.0)

    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Image:         {image_path}")
    print(f"Target Class:  {target_class}")
    print(f"Threshold:     {threshold * 100:.2f}%")
    print("-" * 60)
    
    is_verified = target_prob >= threshold

    if is_verified:
        print(f"[SUCCESS] VERIFIED!")
        print(f"The image is confidently identified as '{target_class}' with {target_prob * 100:.2f}% probability.")
    else:
        print(f"[FAILED] NOT VERIFIED.")
        print(f"The confidence for '{target_class}' is only {target_prob * 100:.2f}%.")
        print(f"The model actually predicts this is '{predicted_class}' with {probabilities[predicted_class] * 100:.2f}% probability.")
        
    print("=" * 60)
    return is_verified

def main():
    parser = argparse.ArgumentParser(description="Verify if an image matches a target monument class")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--class_name", type=str, required=True, help="Target monument class name (e.g., 'tajmahal')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (0.0 to 1.0, default: 0.5)")
    parser.add_argument("--model_type", type=str, default="custom_ensemble_lora", choices=["base", "baseline_lora", "ensemble_lora", "custom_ensemble_lora"], help="Model adapter to use")
    parser.add_argument("--prompt_type", type=str, default="custom_ensemble", choices=["baseline", "custom_ensemble"], help="Prompt strategy to use")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    verify_monument(
        image_path=args.image, 
        target_class=args.class_name, 
        threshold=args.threshold, 
        model_type=args.model_type,
        prompt_type=args.prompt_type
    )

if __name__ == "__main__":
    main()
