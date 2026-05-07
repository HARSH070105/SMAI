import os
from PIL import Image
from inference import load_model, precompute_text_features, run_inference

def main():
    # 1. Define the classes you want to classify
    # Here we use a small subset of the monuments as an example
    monuments = [
        "Ajanta Caves", "Charar-E- Sharif", "Chhota_Imambara", "Ellora Caves", 
        "Gateway of India", "Hawa mahal", "tajmahal"
    ]
    
    # 2. Path to your LoRA adapter weights (if you have them)
    # If None, it will just use the base CLIP model
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    lora_weights_path = os.path.join(ROOT_DIR, "models", "monument_lora_adapters")
    if not os.path.exists(lora_weights_path):
        print(f"LoRA weights not found at {lora_weights_path}. Using base model only.")
        lora_weights_path = None
        
    # 3. Load the model and processor
    model, processor, device = load_model(
        base_model_id="openai/clip-vit-base-patch32",
        lora_weights_path=lora_weights_path
    )
    
    # 4. Precompute text features for your classes
    # This only needs to be done once per session/app startup
    print("Precomputing text features...")
    text_features = precompute_text_features(model, processor, monuments, device)
    
    # 5. Load an image
    # For this example, we'll create a dummy red image
    # In a real scenario, you would do: image = Image.open("path/to/image.jpg").convert("RGB")
    print("Loading test image...")
    # dummy_image = Image.new('RGB', (224, 224), color='red')
    # image = Image.open("/home/bluebottle/Downloads/Taj-Mahal.jpg").convert("RGB")
    image = Image.open("/home/bluebottle/Downloads/chotta.jpg").convert("RGB")


    # 6. Run inference
    print("Running inference...")
    predicted_class, probabilities = run_inference(
        image=image, 
        model=model, 
        processor=processor, 
        text_features=text_features, 
        class_names=monuments, 
        device=device
    )
    
    # 7. Print results
    print("\n" + "="*40)
    print(f"PREDICTED MONUMENT: {predicted_class}")
    print("="*40)
    
    print("\nTop 3 Probabilities:")
    # Get the top 3 highest probabilities
    top_3 = list(probabilities.items())[:3]
    for cls_name, prob in top_3:
        print(f"  {cls_name}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
