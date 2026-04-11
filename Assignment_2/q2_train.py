import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 1. Configuration & Reproducibility
torch.manual_seed(42) # Seed specified in assignment instructions
BATCH_SIZE = 32
EPOCHS = 15 # Train for 15 epochs as per Q2B.1
IMG_SIZE = 128 # Resize all images to 128x128 as per instructions
DATA_DIR = './Dataset' # Point this to the directory shown in your screenshot
MODEL_SAVE_PATH = 'q2_baseline_cnn.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")

# 2. Data Preparation
# Transformations: Resize to 128x128, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("[*] Loading dataset...")
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = full_dataset.classes
print(f"[*] Found {len(full_dataset)} images belonging to {len(class_names)} classes: {class_names}")

# Split into 80% train and 20% test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"[*] Training on {train_size} images, testing on {test_size} images.\n")

# 3. Model Definition (Baseline CNN for Q2B.1)
class BaselineCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCurrencyCNN, self).__init__()
        
        # Helper function to create a Conv Block: Conv2D -> BatchNorm -> ReLU -> MaxPool(2)
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        # 4 Convolutional blocks as specified
        self.features = nn.Sequential(
            conv_block(3, 32),   # Output: 32 x 64 x 64
            conv_block(32, 64),  # Output: 64 x 32 x 32
            conv_block(64, 128), # Output: 128 x 16 x 16
            conv_block(128, 256) # Output: 256 x 8 x 8
        )
        
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Output: 256 x 1 x 1
        
        # Single Linear layer at the end
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = BaselineCurrencyCNN(num_classes=len(class_names)).to(device)

# 4. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
print("[*] Starting training phase...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    train_acc = 100 * correct_train / total_train
    avg_train_loss = running_loss / len(train_loader)
    
    # 6. Evaluation Loop
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    test_acc = 100 * correct_test / total_test
    print(f"=== Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% ===\n")

# 7. Save Model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, MODEL_SAVE_PATH)
print(f"[*] Training complete. Model saved to {MODEL_SAVE_PATH}")