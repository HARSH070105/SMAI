import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

def load_model(base_model_id="openai/clip-vit-base-patch32", lora_weights_path=None, device=None):
    """
    Loads the base CLIP model, processor, and optionally LoRA adapters.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading {base_model_id} on {device}...")
    base_model = CLIPModel.from_pretrained(base_model_id).to(device)
    processor = CLIPProcessor.from_pretrained(base_model_id)
    
    if lora_weights_path and os.path.exists(lora_weights_path):
        print(f"Loading LoRA weights from {lora_weights_path}...")
        model = PeftModel.from_pretrained(base_model, lora_weights_path).to(device)
    else:
        model = base_model
        
    model.eval()
    return model, processor, device

def precompute_text_features(model, processor, class_names, device, use_custom_ensemble=True):
    """
    Precomputes text embeddings for a given set of class names.
    If use_custom_ensemble is True, uses multiple prompts from custom_prompts.py and averages them.
    Otherwise, uses a single "A photo of the {class_name}." prompt.
    """
    if use_custom_ensemble:
        try:
            from custom_prompts import custom_monument_prompts
            prototype_vectors = []
            with torch.no_grad():
                for cls in class_names:
                    matched_sentences = None
                    clean_cls = cls.replace('_', ' ')
                    
                    if cls in custom_monument_prompts:
                        matched_sentences = custom_monument_prompts[cls]
                    elif clean_cls in custom_monument_prompts:
                        matched_sentences = custom_monument_prompts[clean_cls]
                    else:
                        for key, prompts in custom_monument_prompts.items():
                            if key.lower() in cls.lower() or cls.lower() in key.lower():
                                matched_sentences = prompts
                                break
                    
                    if matched_sentences:
                        ensemble_sentences = matched_sentences
                    else:
                        ensemble_sentences = [f"A photo of the {clean_cls}."]
                        
                    text_inputs = processor(text=ensemble_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
                    
                    try:
                        text_outputs = model.get_text_features(**text_inputs)
                    except AttributeError:
                        text_outputs = model.base_model.get_text_features(**text_inputs)
                        
                    if not isinstance(text_outputs, torch.Tensor):
                        if hasattr(text_outputs, "text_embeds"):
                            text_features = text_outputs.text_embeds
                        elif hasattr(text_outputs, "pooler_output"):
                            text_features = text_outputs.pooler_output
                        else:
                            text_features = text_outputs[0]
                    else:
                        text_features = text_outputs
                        
                    # Normalize all vectors
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    # Average them into a single vector
                    mean_vector = text_features.mean(dim=0)
                    # Normalize the final averaged prototype vector
                    prototype_vector = mean_vector / mean_vector.norm(dim=-1, keepdim=True)
                    prototype_vectors.append(prototype_vector)
                    
            # Stack all individual vectors into a single matrix
            text_features = torch.stack(prototype_vectors).to(device)
            return text_features
        except ImportError:
            print("Warning: custom_prompts module not found. Falling back to single prompt baseline.")

    # Fallback / Single Prompt Baseline
    prompts = [f"A photo of the {cls.replace('_', ' ')}." for cls in class_names]
    
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Try using get_text_features, fallback to base_model if peft doesn't delegate it
        try:
            text_outputs = model.get_text_features(**text_inputs)
        except AttributeError:
            text_outputs = model.base_model.get_text_features(**text_inputs)
        
        # Handle if transformers returns an object instead of a tensor
        if not isinstance(text_outputs, torch.Tensor):
            if hasattr(text_outputs, "text_embeds"):
                text_features = text_outputs.text_embeds
            elif hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs[0]
        else:
            text_features = text_outputs
            
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    return text_features

def run_inference(image, model, processor, text_features, class_names, device):
    """
    Runs inference on a single PIL image using precomputed text features.
    Returns the predicted class and a dictionary of probabilities.
    """
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        
        try:
            image_outputs = model.get_image_features(**image_inputs)
        except AttributeError:
            image_outputs = model.base_model.get_image_features(**image_inputs)
            
        # Handle if transformers returns an object instead of a tensor
        if not isinstance(image_outputs, torch.Tensor):
            if hasattr(image_outputs, "image_embeds"):
                image_features = image_outputs.image_embeds
            elif hasattr(image_outputs, "pooler_output"):
                image_features = image_outputs.pooler_output
            else:
                image_features = image_outputs[0]
        else:
            image_features = image_outputs
            
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get predictions
        similarity_scores = similarity_scores.squeeze().cpu().numpy()
        predicted_idx = similarity_scores.argmax()
        predicted_class = class_names[predicted_idx]
        
        # Create a dictionary of class probabilities
        probabilities = {class_names[i]: float(similarity_scores[i]) for i in range(len(class_names))}
        
        # Sort probabilities by value in descending order
        probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        
    return predicted_class, probabilities
