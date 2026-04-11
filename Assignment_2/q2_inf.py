import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

IMG_SIZE = 128
MODEL_SAVE_PATH = 'q2_baseline_cnn.pth'

# 1. Re-define the Model Architecture
# (Must exactly match the training architecture to load weights)
class BaselineCurrencyCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCurrencyCNN, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
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
        x = self.classifier(x)
        return x

def predict_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[!] Error: Model file '{MODEL_SAVE_PATH}' not found. Please train first.")
        return

    # 2. Load the Checkpoint
    print(f"[*] Loading model checkpoint from {MODEL_SAVE_PATH}...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    
    model = BaselineCurrencyCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode (important for BatchNorm)

    # 3. Prepare the Image
    print(f"[*] Processing image: {image_path}")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[!] Error loading image: {e}")
        return

    # Add batch dimension (C, H, W) -> (1, C, H, W)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[predicted_idx.item()]
    
    print("\n" + "="*40)
    print("      INFERENCE RESULT")
    print("="*40)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence:      {confidence.item() * 100:.2f}%")
    print("="*40)
    
    # Optional: Print probabilities for all classes
    print("\nClass Probabilities:")
    for i, class_name in enumerate(class_names):
        print(f" - {class_name}: {probabilities[i].item() * 100:.2f}%")

if __name__ == "__main__":
    # Example usage: python inference.py path/to/image.jpg
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
    else:
        predict_image(sys.argv[1])