"""
dataset_utils.py

Loads image classification datasets from Hugging Face Hub and enriches
class labels with monument metadata from a local JSON file.
"""

import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image


# ---------------------------------------------------------------------------
# 1.  Load raw splits from Hugging Face
# ---------------------------------------------------------------------------

def load_hf_dataset(repo_id: str, image_column: str = "image", label_column: str = "label"):
    """
    Loads a dataset from Hugging Face Hub.
    Expects the repo to have 'train' and 'test' splits.

    Args:
        repo_id      : HuggingFace dataset repo, e.g. "your-username/monuments-dataset"
        image_column : column name that holds the PIL image
        label_column : column name that holds the integer class index

    Returns:
        train_dataset, test_dataset  (HuggingFace Dataset objects)
        class_names                  (list[str], derived from dataset features)
    """
    print(f"Loading dataset '{repo_id}' from Hugging Face Hub...")
    dataset = load_dataset(repo_id)

    train_dataset = dataset["train"]
    test_dataset  = dataset["test"]

    # Extract class names from the ClassLabel feature
    label_feature = train_dataset.features[label_column]
    class_names   = label_feature.names          # list of string labels
    print(f"  Found {len(class_names)} classes, "
          f"{len(train_dataset)} train samples, "
          f"{len(test_dataset)} test samples.")

    return train_dataset, test_dataset, class_names


# ---------------------------------------------------------------------------
# 2.  Load monument metadata JSON
# ---------------------------------------------------------------------------

def load_monument_metadata(json_path: str) -> dict:
    """
    Loads the monument metadata JSON file.

    Args:
        json_path : local path to the JSON file, e.g. "monuments_metadata.json"

    Returns:
        dict keyed by monument name, e.g.
        { "Ajanta Caves": { "history": "...", "fun_facts": [...], ... }, ... }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} monuments from '{json_path}'.")
    return metadata


# ---------------------------------------------------------------------------
# 3.  Build enriched text prompts per class
# ---------------------------------------------------------------------------

def build_enriched_prompts(class_names: list, metadata: dict) -> dict:
    """
    Creates a dictionary of text prompts for each class name.
    Uses metadata fields (history, fun_facts, category, location) when available,
    otherwise falls back to a simple default prompt.

    Args:
        class_names : list of class name strings
        metadata    : dict loaded from load_monument_metadata()

    Returns:
        dict mapping each class name → list[str] of prompt sentences
    """
    prompts = {}

    for cls in class_names:
        # Try direct key match first, then try replacing underscores
        meta = metadata.get(cls) or metadata.get(cls.replace("_", " "))

        if meta is None:
            # Fallback: basic prompt
            prompts[cls] = [f"A photo of the {cls.replace('_', ' ')}."]
            continue

        name      = meta.get("display_name", cls.replace("_", " "))
        location  = meta.get("location", "")
        category  = meta.get("category", "")
        history   = meta.get("history", "")
        fun_facts = meta.get("fun_facts", [])

        sentence_list = [
            f"A photo of {name}.",
            f"An image of the {name}.",
        ]

        if location:
            sentence_list.append(f"A photo of {name}, located in {location}.")

        if category:
            sentence_list.append(f"A photo of {name}, a {category}.")

        if location and category:
            sentence_list.append(
                f"A photo of {name}, a {category} located in {location}."
            )

        # Add a few fun-fact sentences (limit to 3 to avoid very long prompts)
        for fact in fun_facts[:3]:
            sentence_list.append(f"A photo of {name}. {fact}")

        # Prepend the first ~200 chars of history as a context prompt
        if history:
            snippet = history[:200].rstrip()
            sentence_list.append(f"A photo of {name}. {snippet}")

        prompts[cls] = sentence_list

    return prompts


# ---------------------------------------------------------------------------
# 4.  PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class MonumentDataset(Dataset):
    """
    Wraps a HuggingFace dataset split so it works with a PyTorch DataLoader.
    Returns (PIL.Image, int label) pairs.
    """

    def __init__(self, hf_split, image_column: str = "image", label_column: str = "label"):
        self.data         = hf_split
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image  = sample[self.image_column]          # already a PIL.Image from HF
        label  = sample[self.label_column]           # integer index

        # Ensure it's a PIL Image (some datasets return dicts or file paths)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        return image, label
