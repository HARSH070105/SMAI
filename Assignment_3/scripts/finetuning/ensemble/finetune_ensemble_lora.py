import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 5e-5

TRAIN_DIR = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "train")

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
    print(f"Loading {model_id} on {device}...")

    base_model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_processor.image_mean, 
            std=processor.image_processor.image_std
        )
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print("Encoding ensemble text prompts...")
    prototype_vectors = []
    
    with torch.no_grad():
        for monument in tqdm(class_names, desc="Processing Prototypes"):
            clean_name = monument.replace('_', ' ')
            sentences = [template.format(clean_name) for template in ensemble_templates]
            
            text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
            text_outputs = base_model.get_text_features(**text_inputs)
            
            if not isinstance(text_outputs, torch.Tensor):
                text_features = getattr(text_outputs, "text_embeds", getattr(text_outputs, "pooler_output", text_outputs[0]))
            else:
                text_features = text_outputs
                
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_vector = text_features.mean(dim=0)
            prototype_vector = mean_vector / mean_vector.norm(dim=-1, keepdim=True)
            prototype_vectors.append(prototype_vector)
            
    text_prototypes = torch.stack(prototype_vectors).to(device)
    text_prototypes.requires_grad = False

    config = LoraConfig(
        r=16,                   
        lora_alpha=16,          
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    logit_scale = model.base_model.model.logit_scale.exp().item()

    print("\nStarting LoRA Fine-Tuning (Ensemble Prompts)...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            image_features = model.base_model.model.get_image_features(pixel_values=images)
            
            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(image_features, "image_embeds", getattr(image_features, "pooler_output", image_features[0]))
                
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = logit_scale * image_features @ text_prototypes.T
            
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            loop.set_postfix(loss=loss.item(), acc=correct/total_samples)

        print(f"Epoch {epoch+1} Summary: Avg Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/total_samples * 100:.2f}%")

    SAVE_PATH = os.path.join(ROOT_DIR, "models", "ensemble_lora_adapters")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    print(f"\nTraining Complete! LoRA adapters saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
