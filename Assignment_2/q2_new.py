import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy

# --- HPC Logger ---
class HPCLogger(object):
    def __init__(self, filename="Logs/super_model_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.log.flush()

sys.stdout = HPCLogger()

# --- Configuration ---
torch.manual_seed(42)
BATCH_SIZE = 32
MAX_EPOCHS = 100     # Let it run much longer
PATIENCE = 12        # Stop if no improvement for 12 epochs
IMG_SIZE = 128
DATA_DIR = './Dataset' 
MODEL_SAVE_PATH = 'Models/super_model_cnn.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")
print("[*] Engaging Unconstrained Super Model Training...")

# --- 1. Aggressive Data Augmentation ---
# We force the model to look at the note, not the background, by heavily 
# distorting the color, angle, and position during training.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(45), # Increased rotation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Added Saturation & Hue shifts
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2), # Currency notes can be upside down
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)), # Added zooming (scale)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Optional but powerful: Random Erasing blocks out random chunks of the image, 
    # forcing the model to learn multiple features (e.g., if the '50' is covered, it MUST use Gandhi's face)
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)) 
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Data Loading ---
full_dataset = datasets.ImageFolder(root=DATA_DIR)
class_names = full_dataset.classes
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Apply different transforms securely
train_dataset.dataset = copy.copy(full_dataset)
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- 2. Advanced Architecture ---
class SuperCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super(SuperCurrencyCNN, self).__init__()
        def conv_block(in_c, out_c, drop_rate=0.0):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False), # No bias needed before BatchNorm
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True), # LeakyReLU prevents "dead" neurons
                nn.MaxPool2d(kernel_size=2, stride=2)
            ]
            if drop_rate > 0:
                layers.append(nn.Dropout2d(drop_rate)) # Spatial dropout drops entire feature maps
            return nn.Sequential(*layers)
            
        # Deeper network: 5 blocks instead of 4, with increasing channel capacity
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128, drop_rate=0.1),
            conv_block(128, 256, drop_rate=0.2),
            conv_block(256, 512, drop_rate=0.3) # Output: 512 x 4 x 4
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Heavy dropout on the dense classifier to force robust learning
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = SuperCurrencyCNN(num_classes=len(class_names)).to(device)

# --- 3. Optimizer & Dynamic Scheduling ---
# AdamW decouples weight decay from the gradient update, providing much better L2 regularization
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)

# If the validation accuracy stops improving for 4 epochs, cut the learning rate in half
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)
criterion = nn.CrossEntropyLoss()

# --- Training Loop with Early Stopping ---
best_test_acc = 0.0
epochs_no_improve = 0

print(f"\n[*] Starting Training (Max Epochs: {MAX_EPOCHS})")
for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    
    for images, labels in train_loader:
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
        
    model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    train_acc = 100 * correct_train / total_train
    test_acc = 100 * correct_test / total_test
    avg_train_loss = running_loss / len(train_loader)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1:03d}/{MAX_EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Step the scheduler based on Test Accuracy
    scheduler.step(test_acc)
    
    # Early Stopping & Checkpointing
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        epochs_no_improve = 0
        torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, MODEL_SAVE_PATH)
        print("  -> New best model saved!")
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= PATIENCE:
        print(f"\n[!] Early Stopping Triggered! Test accuracy hasn't improved for {PATIENCE} epochs.")
        break

print(f"\n[*] Training Complete. Best Test Accuracy: {best_test_acc:.2f}%")