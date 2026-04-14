import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================================
# 🔧 USER CONFIG
# ================================
IMAGE_FOLDER = "Test_Notes_Augmented" 
OUTPUT_TEXT_FILE = "Logs/evaluation_results.txt"
MODEL_PATHS = [
    "Models/super_model_cnn.pth",
    "Models/q2_baseline_cnn.pth",
    "Models/q2_mobile_cnn.pth",
    "Models/q2_mobile_aug_cnn.pth"
]
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ================================


# -------------------------
# 1. MODEL DEFINITIONS
# -------------------------

# --- Baseline Architecture ---
class BaselineCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- Mobile Architecture ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MobileCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def standard_conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
        def separable_conv_block(in_c, out_c):
            return nn.Sequential(
                DepthwiseSeparableConv(in_c, out_c),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
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

# --- Super Model Architecture ---
class SuperCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super(SuperCurrencyCNN, self).__init__()
        def conv_block(in_c, out_c, drop_rate=0.0):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ]
            if drop_rate > 0:
                layers.append(nn.Dropout2d(drop_rate))
            return nn.Sequential(*layers)
            
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128, drop_rate=0.1),
            conv_block(128, 256, drop_rate=0.2),
            conv_block(256, 512, drop_rate=0.3)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -------------------------
# 2. LOAD MODELS
# -------------------------
models = []
class_names = None

print(f"[*] Initializing inference on {DEVICE}...")

for model_path in MODEL_PATHS:
    if not os.path.exists(model_path):
        print(f"[!] Warning: '{model_path}' not found. Skipping.")
        continue

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if class_names is None:
        class_names = ckpt["class_names"]

    num_classes = len(ckpt["class_names"])

    # Auto-route to the correct architecture class based on filename
    if "super" in model_path.lower():
        model = SuperCurrencyCNN(num_classes)
    elif "mobile" in model_path.lower():
        model = MobileCurrencyCNN(num_classes)
    else:
        model = BaselineCurrencyCNN(num_classes)

    try:
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        models.append((os.path.basename(model_path), model))
        print(f"[*] Successfully loaded {model_path}")
    except Exception as e:
        print(f"[!] Failed to load {model_path}: {e}")


# -------------------------
# 3. TRANSFORM
# -------------------------
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -------------------------
# 4. INFERENCE & TRACKING
# -------------------------
if not models:
    print("[!] No models loaded. Exiting.")
    exit()

# Dictionaries to track true vs predicted labels for each model
y_true_all = {model_name: [] for model_name, _ in models}
y_pred_all = {model_name: [] for model_name, _ in models}

print(f"\n[*] Scanning '{IMAGE_FOLDER}' for images...")

# Open the text file to write individual predictions
with open(OUTPUT_TEXT_FILE, "w") as file:
    file.write("=== INDIVIDUAL IMAGE PREDICTIONS ===\n\n")
    
    for img_name in os.listdir(IMAGE_FOLDER):

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        
        # Extract true label assuming format like "10_1.jpeg" -> "10"
        # Format extracted number to match model classes ("10" -> "Notes_10")
        label_number = img_name.split("_")[0]
        true_label = f"Notes_{label_number}"

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[!] Could not open {img_name}: {e}")
            continue
        
        # --- Dynamic Center Crop ---
        min_dim = min(img.size)
        img = transforms.CenterCrop(min_dim)(img)
        # --------------------------------

        img_tensor = base_transform(img).unsqueeze(0).to(DEVICE)

        # Print to console and write to file
        print(f"\n{img_name} (True: {true_label})")
        file.write(f"{img_name} (True: {true_label})\n")

        for model_name, model in models:
            with torch.no_grad():
                out = model(img_tensor)
                pred_idx = torch.argmax(out, dim=1).item()
            
            pred_label = class_names[pred_idx]
            
            # Store for metrics
            y_true_all[model_name].append(true_label)
            y_pred_all[model_name].append(pred_label)

            print(f"  {model_name}: {pred_label}")
            file.write(f"  {model_name}: {pred_label}\n")
        
        file.write("\n")

print(f"\n[*] Individual predictions saved to {OUTPUT_TEXT_FILE}")

# -------------------------
# 5. METRICS & CONFUSION MATRICES
# -------------------------
print("\n[*] Calculating Metrics and Generating Confusion Matrices...")

# Ensure class_names are strings for the scikit-learn functions
str_class_names = [str(c) for c in class_names]

# Append overall metrics to the text file
with open(OUTPUT_TEXT_FILE, "a") as file:
    file.write("\n\n" + "="*40 + "\n")
    file.write("=== MODEL PERFORMANCE METRICS ===\n")
    file.write("="*40 + "\n\n")

    for model_name, _ in models:
        y_true = y_true_all[model_name]
        y_pred = y_pred_all[model_name]

        # 1. Generate Classification Report (Precision, Recall, F1, Accuracy)
        report = classification_report(y_true, y_pred, labels=str_class_names, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        # Write to file
        file.write(f"--- {model_name} ---\n")
        file.write(f"Accuracy: {acc:.4f}\n\n")
        file.write(report)
        file.write("\n" + "-"*40 + "\n\n")
        
        # Print to console as well
        print(f"\n--- {model_name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)

        # 2. Generate and Save Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred, labels=str_class_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=str_class_names, yticklabels=str_class_names)
        
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Clean up filename for the image
        safe_model_name = model_name.replace(".pth", "")
        cm_filename = f"Logs/cm_{safe_model_name}.png"
        
        plt.tight_layout()
        plt.savefig(cm_filename)
        plt.close() # Close the figure to free up memory
        
        print(f"[*] Saved confusion matrix image: {cm_filename}")

print(f"\n[*] All done! Full evaluation report appended to {OUTPUT_TEXT_FILE}")