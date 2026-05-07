import os

scripts = ["identify_monuments.py", "identify_monuments_ensemble.py", "identify_monuments_custom_ensemble.py"]

for script in scripts:
    with open(script, "r") as f:
        content = f.read()
    
    old_code = """model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()"""
    
    new_code = """base_model = CLIPModel.from_pretrained(model_id).to(device)
from peft import PeftModel
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model = PeftModel.from_pretrained(base_model, os.path.join(ROOT_DIR, "models", "monument_lora_adapters")).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()"""
    
    content = content.replace(old_code, new_code)
    
    # Reports
    if "custom_ensemble" in script:
        content = content.replace("classification_report_custom_ensemble.txt", "classification_report_custom_ensemble_lora.txt")
        content = content.replace("confusion_matrix_custom_ensemble.png", "confusion_matrix_custom_ensemble_lora.png")
    elif "ensemble" in script:
        content = content.replace("classification_report_ensemble.txt", "classification_report_ensemble_lora.txt")
        content = content.replace("confusion_matrix_ensemble.png", "confusion_matrix_ensemble_lora.png")
    else:
        content = content.replace("classification_report_baseline.txt", "classification_report_baseline_lora.txt")
        content = content.replace("confusion_matrix_baseline.png", "confusion_matrix_baseline_lora.png")
    
    # Method calls
    content = content.replace("model.get_text_features", "base_model.get_text_features")
    content = content.replace("model.get_image_features", "base_model.get_image_features")
    
    new_script_name = script.replace(".py", "_lora.py")
    with open(new_script_name, "w") as f:
        f.write(content)
    print(f"Created {new_script_name}")
