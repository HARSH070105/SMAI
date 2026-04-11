import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --- Configuration ---
torch.manual_seed(42)
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = 128
DATA_DIR = './Dataset' 
MODEL_SAVE_PATH = 'q2_mobile_cnn.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")

# --- Data Loading ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Define Depthwise Separable Convolution ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise conv: groups=in_channels applies a single filter per input channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # Pointwise conv: 1x1 convolution to mix the channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# --- Define Mobile-Friendly CNN ---
class MobileCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MobileCurrencyCNN, self).__init__()
        
        def standard_conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
        def separable_conv_block(in_c, out_c):
            return nn.Sequential(
                DepthwiseSeparableConv(in_c, out_c),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        # First two blocks use standard convolutions
        self.block1 = standard_conv_block(3, 32)
        self.block2 = standard_conv_block(32, 64)
        
        # Last two blocks use depthwise separable convolutions
        self.block3 = separable_conv_block(64, 128)
        self.block4 = separable_conv_block(128, 256)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- Helper to calculate parameters ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Recreate Baseline to compare parameters
class BaselineCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
            )
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

# Initialize models and print parameter comparison (Q2B.2.a)
num_classes = len(full_dataset.classes)
baseline_model = BaselineCurrencyCNN(num_classes)
mobile_model = MobileCurrencyCNN(num_classes).to(device)

base_params = count_parameters(baseline_model)
mobile_params = count_parameters(mobile_model)

print("\n" + "="*40)
print("      PARAMETER COMPARISON")
print("="*40)
print(f"Baseline CNN Parameters: {base_params:,}")
print(f"Mobile CNN Parameters:   {mobile_params:,}")
print(f"Compression Ratio:       {base_params/mobile_params:.2f}x")
print("="*40 + "\n")

# --- Training Loop ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobile_model.parameters(), lr=0.001)

print("[*] Starting training phase for Mobile CNN...")
for epoch in range(EPOCHS):
    mobile_model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = mobile_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    # Validation Loop
    mobile_model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = mobile_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    train_acc = 100 * correct_train / total_train
    test_acc = 100 * correct_test / total_test
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

torch.save({'model_state_dict': mobile_model.state_dict(), 'class_names': full_dataset.classes}, MODEL_SAVE_PATH)
print(f"\n[*] Training complete. Mobile model saved to {MODEL_SAVE_PATH}")