import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class HPCLogger(object):
    def __init__(self, filename="Logs/q2_aug_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.log.flush()

sys.stdout = HPCLogger()

torch.manual_seed(42)
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = 128
DATA_DIR = './Dataset' 
MODEL_SAVE_PATH = 'Models/q2_mobile_aug_cnn.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")

# --- Data Augmentation Pipeline (Q2B.3) ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and split
full_dataset = datasets.ImageFolder(root=DATA_DIR)
class_names = full_dataset.classes
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Apply specific transforms
train_dataset.dataset = copy.copy(full_dataset)
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Definition ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MobileCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MobileCurrencyCNN, self).__init__()
        def standard_conv_block(in_c, out_c):
            return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        def separable_conv_block(in_c, out_c):
            return nn.Sequential(DepthwiseSeparableConv(in_c, out_c), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.block1 = standard_conv_block(3, 32)
        self.block2 = standard_conv_block(32, 64)
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
        return self.classifier(x)

mobile_aug_model = MobileCurrencyCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobile_aug_model.parameters(), lr=0.001)

print("\n[*] Starting Data Augmentation Training (15 Epochs)...")
for epoch in range(EPOCHS):
    mobile_aug_model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = mobile_aug_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    mobile_aug_model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = mobile_aug_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    train_acc = 100 * correct_train / total_train
    test_acc = 100 * correct_test / total_test
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

torch.save({'model_state_dict': mobile_aug_model.state_dict(), 'class_names': class_names}, MODEL_SAVE_PATH)
print(f"[*] Training complete. Augmented model saved to {MODEL_SAVE_PATH}")
