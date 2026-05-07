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

def main():
    print(f"Loading {model_id} on {device}...")

    # ==========================================
    # 2. Load Model, Processor
    # ==========================================
    base_model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # ==========================================
    # 3. Data Loading & Augmentations
    # ==========================================
    # Aggressive augmentations help prevent background bias
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Zoom in randomly
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_processor.image_mean, 
            std=processor.image_processor.image_std
        )
    ])

    # Use ImageFolder. It loads classes in ALPHABETICAL ORDER.
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # ==========================================
    # 4. Generate Text Prototypes (Baseline)
    # ==========================================
    print("Encoding baseline text prompts...")
    baseline_prompts = [f"A photo of the {monument.replace('_', ' ')}." for monument in class_names]
    
    with torch.no_grad():
        text_inputs = processor(text=baseline_prompts, return_tensors="pt", padding=True).to(device)
        text_outputs = base_model.get_text_features(**text_inputs)
        
        if not isinstance(text_outputs, torch.Tensor):
            if hasattr(text_outputs, "text_embeds"):
                text_features = text_outputs.text_embeds
            elif hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs[0]
        else:
            text_features = text_outputs
            
        text_prototypes = text_features / text_features.norm(dim=-1, keepdim=True)
        text_prototypes.requires_grad = False

    # ==========================================
    # 5. Apply LoRA to the Vision Encoder
    # ==========================================
    # We only want to inject trainable adapters into the attention blocks of the vision model.
    config = LoraConfig(
        r=32,                   # Rank of the update matrices (controls parameter count)
        lora_alpha=16,          # Scaling factor
        target_modules=["q_proj", "v_proj"], # Target the Query and Value attention layers
        lora_dropout=0.1,
        bias="none",
    )

    # Apply LoRA. The peft library will automatically freeze the base model 
    # and only make the injected LoRA matrices trainable.
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()

    # ==========================================
    # 6. Training Loop
    # ==========================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # CLIP uses a temperature scaling parameter to sharpen logits. We extract it here.
    logit_scale = model.base_model.model.logit_scale.exp().item()

    print("\nStarting LoRA Fine-Tuning (Baseline Prompts)...")

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
            
            # 1. Forward pass: Get vision embeddings through our LoRA-adapted model
            image_features = model.base_model.model.get_image_features(pixel_values=images)
            
            # FIX: Ensure it's a tensor (handles huggingface version differences)
            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(image_features, "image_embeds", getattr(image_features, "pooler_output", image_features[0]))
                
            # 2. Normalize the image features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 3. Calculate logits (Cosine Similarity * Logit Scale)
            # Dot product of Image(Batch, 512) and Text(512, Num_Classes) = Logits(Batch, Num_Classes)
            logits = logit_scale * image_features @ text_prototypes.T
            
            # 4. Calculate Loss & Backpropagate
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy for monitoring
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            loop.set_postfix(loss=loss.item(), acc=correct/total_samples)

        print(f"Epoch {epoch+1} Summary: Avg Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/total_samples * 100:.2f}%")

    # ==========================================
    # 7. Save the LoRA Adapters
    # ==========================================
    SAVE_PATH = os.path.join(ROOT_DIR, "models", "baseline_lora_adapters")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    print(f"\nTraining Complete! LoRA adapters saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
