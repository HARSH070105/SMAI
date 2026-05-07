import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_SCRIPT = os.path.join(ROOT_DIR, "scripts", "evaluation", "evaluate.py")

model_types = ["base", "baseline_lora", "ensemble_lora", "custom_ensemble_lora"]
prompt_types = ["baseline", "ensemble", "custom_ensemble"]

for model in model_types:
    # Check if adapter exists for LoRA models
    if model != "base":
        adapter_path = os.path.join(ROOT_DIR, "models", f"{model}_adapters")
        if not os.path.exists(adapter_path):
            print(f"Skipping model '{model}' because adapter was not found at {adapter_path}")
            continue

    for prompt in prompt_types:
        print(f"\n{'='*60}")
        print(f"Running Evaluation for Model: {model} | Prompt: {prompt}")
        print(f"{'='*60}\n")
        
        cmd = [
            "python", EVAL_SCRIPT,
            "--model_type", model,
            "--prompt_type", prompt
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for Model: {model} | Prompt: {prompt}")
            print(f"Error: {e}")

print("\nMatrix Evaluation Complete!")
