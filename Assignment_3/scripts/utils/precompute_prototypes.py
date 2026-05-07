import os
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from custom_prompts import custom_monument_prompts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

print(f"Loading {model_id} on {device}...")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

monuments_24 = [
    "Ajanta Caves", "Charar-E- Sharif", "Chhota_Imambara", "Ellora Caves", 
    "Fatehpur Sikri", "Gateway of India", "Hawa mahal", "Humayun_s Tomb", 
    "India_gate", "Khajuraho", "Sun Temple Konark", "alai_darwaza", 
    "alai_minar", "basilica_of_bom_jesus", "charminar", "golden temple", 
    "iron_pillar", "jamali_kamali_tomb", "lotus_temple", "mysore_palace", 
    "qutub_minar", "tajmahal", "tanjavur temple", "victoria memorial"
]

prototype_dict = {}

print("Encoding and averaging custom ensemble text prompts...")
with torch.no_grad():
    for monument in tqdm(monuments_24, desc="Processing Text Prototypes"):
        ensemble_sentences = custom_monument_prompts[monument]
        
        # Process and encode
        text_inputs = processor(text=ensemble_sentences, return_tensors="pt", padding=True).to(device)
        text_outputs = model.get_text_features(**text_inputs)
        
        # Extract the tensor if transformers returned an object
        if not isinstance(text_outputs, torch.Tensor):
            if hasattr(text_outputs, "text_embeds"):
                text_features = text_outputs.text_embeds
            elif hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs[0]
        else:
            text_features = text_outputs
        
        # Math: 1. Normalize all vectors
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Math: 2. Average them into a single vector
        mean_vector = text_features.mean(dim=0)
        # Math: 3. Normalize the final averaged prototype vector
        prototype_vector = mean_vector / mean_vector.norm(dim=-1, keepdim=True)
        
        prototype_dict[monument] = prototype_vector.cpu()

save_path = os.path.join(ROOT_DIR, "models", "custom_monument_24_prototypes.pt")
torch.save(prototype_dict, save_path)
print(f"Saved prototype dictionary to {save_path}")
