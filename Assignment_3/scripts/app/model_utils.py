"""
model_utils.py  (updated)

Loads CLIP + optional LoRA weights, computes text embeddings using
either custom_prompts.py or monument metadata, and runs inference.
"""

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel


# ---------------------------------------------------------------------------
# 1.  Model loading
# ---------------------------------------------------------------------------

def load_model(
    base_model_id: str = "openai/clip-vit-base-patch32",
    lora_weights_path: str = None,
    device: str = None,
):
    """
    Loads the base CLIP model, processor, and optionally LoRA adapters.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {base_model_id} on {device}...")
    base_model = CLIPModel.from_pretrained(base_model_id).to(device)
    processor  = CLIPProcessor.from_pretrained(base_model_id)

    if lora_weights_path and os.path.exists(lora_weights_path):
        print(f"Loading LoRA weights from {lora_weights_path}...")
        model = PeftModel.from_pretrained(base_model, lora_weights_path).to(device)
    else:
        model = base_model

    model.eval()
    return model, processor, device


# ---------------------------------------------------------------------------
# 2.  Internal helper: encode a list of text strings → normalised tensor
# ---------------------------------------------------------------------------

def _encode_texts(model, processor, sentences: list, device: str) -> torch.Tensor:
    """
    Tokenises `sentences`, runs through the CLIP text encoder, and returns
    a (N, D) normalised float tensor on `device`.
    """
    text_inputs = processor(
        text=sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        try:
            outputs = model.get_text_features(**text_inputs)
        except AttributeError:                      # PEFT wrapping
            outputs = model.base_model.get_text_features(**text_inputs)

    # Unwrap if the model returned a dataclass / named-tuple
    if not isinstance(outputs, torch.Tensor):
        if hasattr(outputs, "text_embeds"):
            features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs[0]
    else:
        features = outputs

    features = features / features.norm(dim=-1, keepdim=True)
    return features


# ---------------------------------------------------------------------------
# 3.  Text feature pre-computation (three strategies, same API)
# ---------------------------------------------------------------------------

def precompute_text_features(
    model,
    processor,
    class_names: list,
    device: str,
    # --- NEW: pass enriched prompts built from monument metadata ---
    enriched_prompts: dict = None,
    # --- existing flag: fall back to custom_prompts.py ensemble ---
    use_custom_ensemble: bool = True,
) -> torch.Tensor:
    """
    Precomputes one prototype vector per class.

    Priority order:
      1. enriched_prompts  - dict built by dataset_utils.build_enriched_prompts()
      2. custom_prompts.py - existing per-class ensemble
      3. single-prompt baseline

    Args:
        model, processor, class_names, device : as before
        enriched_prompts : dict mapping class_name -> list[str] of sentences,
                           produced by dataset_utils.build_enriched_prompts()
        use_custom_ensemble : if True and enriched_prompts is None,
                              fall back to custom_prompts.py

    Returns:
        (C, D) float tensor of normalised prototype vectors, one per class
    """

    # ------------------------------------------------------------------
    # Strategy 1: metadata-enriched prompts  (NEW)
    # ------------------------------------------------------------------
    if enriched_prompts is not None:
        print("Computing text features using monument metadata prompts...")
        prototype_vectors = []
        for cls in class_names:
            sentences = enriched_prompts.get(
                cls,
                [f"A photo of the {cls.replace('_', ' ')}."],
            )
            features = _encode_texts(model, processor, sentences, device)   # (N, D)
            mean_vec = features.mean(dim=0)
            prototype = mean_vec / mean_vec.norm()
            prototype_vectors.append(prototype)
        return torch.stack(prototype_vectors).to(device)

    # ------------------------------------------------------------------
    # Strategy 2: custom_prompts.py ensemble  (unchanged from original)
    # ------------------------------------------------------------------
    if use_custom_ensemble:
        try:
            from custom_prompts import custom_monument_prompts
            print("Computing text features using custom_prompts.py ensemble...")
            prototype_vectors = []
            for cls in class_names:
                sentences = custom_monument_prompts.get(
                    cls,
                    [f"A photo of the {cls.replace('_', ' ')}."],
                )
                features  = _encode_texts(model, processor, sentences, device)
                mean_vec  = features.mean(dim=0)
                prototype = mean_vec / mean_vec.norm()
                prototype_vectors.append(prototype)
            return torch.stack(prototype_vectors).to(device)
        except ImportError:
            print("Warning: custom_prompts module not found. "
                  "Falling back to single-prompt baseline.")

    # ------------------------------------------------------------------
    # Strategy 3: single-prompt baseline
    # ------------------------------------------------------------------
    print("Computing text features using single-prompt baseline...")
    prompts = [f"A photo of the {cls.replace('_', ' ')}." for cls in class_names]
    features = _encode_texts(model, processor, prompts, device)   # (C, D)
    return features


# ---------------------------------------------------------------------------
# 4.  Inference  (unchanged API)
# ---------------------------------------------------------------------------

def run_inference(image, model, processor, text_features, class_names, device):
    """
    Runs inference on a single PIL image using precomputed text features.

    Args:
        image         : PIL.Image
        model         : loaded CLIP / PEFT model
        processor     : CLIPProcessor
        text_features : (C, D) tensor from precompute_text_features()
        class_names   : list[str]
        device        : str

    Returns:
        predicted_class : str
        probabilities   : dict {class_name: float}, sorted descending
    """
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)

        try:
            img_out = model.get_image_features(**image_inputs)
        except AttributeError:
            img_out = model.base_model.get_image_features(**image_inputs)

        if not isinstance(img_out, torch.Tensor):
            if hasattr(img_out, "image_embeds"):
                image_features = img_out.image_embeds
            elif hasattr(img_out, "pooler_output"):
                image_features = img_out.pooler_output
            else:
                image_features = img_out[0]
        else:
            image_features = img_out

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        scores     = similarity.squeeze().cpu().numpy()

    predicted_idx   = scores.argmax()
    predicted_class = class_names[predicted_idx]
    probabilities   = {
        class_names[i]: float(scores[i]) for i in range(len(class_names))
    }
    probabilities = dict(
        sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    )

    return predicted_class, probabilities
