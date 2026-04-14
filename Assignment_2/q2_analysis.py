import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================================
# 🔧 CONFIGURATION
# ================================
torch.manual_seed(42)
LOG_FILE = "Logs/gradcam_predictions_log.txt"

MODEL_PATHS = [
    "Models/super_model_cnn.pth",
    "Models/q2_baseline_cnn.pth",
    "Models/q2_mobile_cnn.pth",
    "Models/q2_mobile_aug_cnn.pth"
]

# The specific images you want to test and visualize
TARGET_IMAGES = [
    "Test_Notes_Augmented/50_4_orig.jpeg", 
    "Test_Notes_Augmented/500_3_rot180.jpeg",
    "Test_Notes_Augmented/100_3_orig.jpeg"
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================
# 1. MODEL ARCHITECTURES
# ================================

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


# ================================
# 2. HELPER FUNCTIONS
# ================================
def get_gradcam_target_layer(model, model_name):
    """Finds the correct last convolutional layer based on the architecture."""
    if "super" in model_name.lower():
        # Super model: 5th block, 1st layer (Conv2d)
        return [model.features[4][0]]
    elif "mobile" in model_name.lower():
        # Mobile model: 4th block, 1st layer (DepthwiseSeparableConv)
        return [model.block4[0]]
    else:
        # Baseline model: 4th block, 1st layer (Conv2d)
        return [model.features[3][0]]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ================================
# 3. MAIN EXECUTION
# ================================
if __name__ == "__main__":
    
    loaded_models = []
    class_names = None

    print(f"[*] Initializing on {DEVICE}...")

    # Load all 4 models
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"[!] Missing {path}. Skipping.")
            continue
            
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        
        if class_names is None:
            class_names = ckpt["class_names"]
            
        num_classes = len(class_names)
        
        # Route to correct class
        if "super" in path.lower():
            model = SuperCurrencyCNN(num_classes)
        elif "mobile" in path.lower():
            model = MobileCurrencyCNN(num_classes)
        else:
            model = BaselineCurrencyCNN(num_classes)
            
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        loaded_models.append((os.path.basename(path), model))
        print(f"[*] Loaded {os.path.basename(path)}")

    if not loaded_models:
        print("[!] No models loaded. Exiting.")
        exit()

    print(f"\n[*] Processing Images & Generating Grad-CAMs...")

    # Open text file for logging
    with open(LOG_FILE, "w") as log:
        log.write("=== GRAD-CAM IMAGE PREDICTIONS ===\n\n")

        # Iterate through target images
        for img_path in TARGET_IMAGES:
            if not os.path.exists(img_path):
                print(f"[!] Image {img_path} not found. Skipping.")
                continue

            img_name = os.path.basename(img_path)
            
            # Extract true label (e.g., "100_1.jpeg" -> "Notes_100")
            true_label_num = img_name.split('_')[0]
            true_label_str = f"Notes_{true_label_num}"

            print(f"\nEvaluating: {img_name} (True: {true_label_str})")
            log.write(f"{img_name} (True: {true_label_str})\n")

            # Load and prep image
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((128, 128))
            rgb_img = np.float32(img_resized) / 255.0
            input_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)

            # Test image on every model
            for model_name, model in loaded_models:
                safe_model_name = model_name.replace('.pth', '')
                
                # 1. Prediction Logging
                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred_idx = torch.argmax(outputs, dim=1).item()
                    pred_label = class_names[pred_idx]
                
                print(f"  {safe_model_name} predicted: {pred_label}")
                log.write(f"  {safe_model_name}: {pred_label}\n")

                # 2. Grad-CAM Generation
                target_layers = get_gradcam_target_layer(model, safe_model_name)
                
                # Initialize CAM
                cam = GradCAM(model=model, target_layers=target_layers)
                
                # Generate mask
                grayscale_cam = cam(input_tensor=input_tensor)[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
                # Plot and Save
                plt.figure(figsize=(5, 5))
                plt.imshow(visualization)
                plt.axis('off')
                
                # Add text to the image so you know what you are looking at
                plt.title(f"{safe_model_name}\nPred: {pred_label}", fontsize=10)
                
                save_name = f"Logs/gradcam_{safe_model_name}_{img_name.split('.')[0]}.png"
                plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
                plt.close() # Free up memory
                
            log.write("\n")

    print(f"\n[*] All done! Predictions saved to {LOG_FILE} and heatmaps saved as PNGs.")