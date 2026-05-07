"""
Builds data/24_monuments/ from the raw HuggingFace download.

Fixes applied:
  - 'hawa mahal pics' (train) → canonical 'Hawa mahal'
  - 'India gate pics' (train) → canonical 'India_gate'
  - 'lotus_temple' has 0 test images → carves 5 out of train for test

Usage (from repo root):
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --src /path/to/Monuments_SMAI/folder
"""

import os
import shutil
import argparse
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default source: sister directory of the repo (where HuggingFace download lives)
DEFAULT_SRC = os.path.join(
    os.path.dirname(ROOT_DIR),
    "Monuments_SMAI_local", "Amisha2905", "Monuments_SMAI"
)

# Destination expected by all model scripts
DST = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images")

# ==========================================
# 1. Canonical class names
#    Must match monuments_24 in identify_monuments.py exactly
# ==========================================

CANONICAL_CLASSES = [
    "Ajanta Caves", "Charar-E- Sharif", "Chhota_Imambara", "Ellora Caves",
    "Fatehpur Sikri", "Gateway of India", "Hawa mahal", "Humayun_s Tomb",
    "India_gate", "Khajuraho", "Sun Temple Konark", "alai_darwaza",
    "alai_minar", "basilica_of_bom_jesus", "charminar", "golden temple",
    "iron_pillar", "jamali_kamali_tomb", "lotus_temple", "mysore_palace",
    "qutub_minar", "tajmahal", "tanjavur temple", "victoria memorial",
]

# Source folder names that differ from canonical → canonical name
FOLDER_RENAMES = {
    "hawa mahal pics": "Hawa mahal",
    "India gate pics": "India_gate",
}

# Classes to carve a test split from train (class → n images to move to test)
# Only applies when test count is below this number after copying
CARVE_TEST = {
    "lotus_temple": 5,   # 0 test images in the download
    "charminar":    30,  # only 1 test image in the download
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ==========================================
# 2. Helpers
# ==========================================

def image_files(folder: str):
    """Return sorted list of image paths in a folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )


def copy_images(src_folder: str, dst_folder: str, start_idx: int = 1) -> int:
    """
    Copy all images from src_folder to dst_folder, renaming sequentially
    from start_idx to avoid filename collisions when merging folders.
    Returns the next available index.
    """
    os.makedirs(dst_folder, exist_ok=True)
    imgs = image_files(src_folder)
    idx = start_idx
    for img in imgs:
        dst_path = os.path.join(dst_folder, f"{idx}{img.suffix.lower()}")
        shutil.copy2(str(img), dst_path)
        idx += 1
    return idx


# ==========================================
# 3. Main
# ==========================================

def main(src_root: str):
    if not os.path.isdir(src_root):
        print(f"ERROR: Source not found:\n  {src_root}")
        print("Pass the correct path via --src")
        return

    print(f"Source : {src_root}")
    print(f"Dest   : {DST}")
    print()

    summary = []  # (class, train_count, test_count)

    for split in ("train", "test"):
        src_split = os.path.join(src_root, split)
        dst_split = os.path.join(DST, split)

        # Collect all source folders for this split
        if not os.path.isdir(src_split):
            print(f"WARNING: split folder not found: {src_split}")
            continue

        src_folders = {f.name: f.path for f in os.scandir(src_split) if f.is_dir()}

        for canonical in CANONICAL_CLASSES:
            dst_folder = os.path.join(dst_split, canonical)

            # Determine which source folder(s) map to this canonical class
            # A class may appear under its own name or a renamed variant
            sources = []
            if canonical in src_folders:
                sources.append(src_folders[canonical])
            for src_name, canon in FOLDER_RENAMES.items():
                if canon == canonical and src_name in src_folders:
                    sources.append(src_folders[src_name])

            idx = 1
            for src_folder in sources:
                idx = copy_images(src_folder, dst_folder, start_idx=idx)

            count = idx - 1
            if split == "train":
                summary.append([canonical, count, 0])
            else:
                for row in summary:
                    if row[0] == canonical:
                        row[2] = count
                        break

    # ==========================================
    # 4. Carve test splits for problem classes
    # ==========================================
    for canonical, n_carve in CARVE_TEST.items():
        test_folder = os.path.join(DST, "test", canonical)
        train_folder = os.path.join(DST, "train", canonical)
        existing_test = len(image_files(test_folder))

        if existing_test >= n_carve:
            continue  # already has enough test images

        train_imgs = image_files(train_folder)
        to_move = train_imgs[:n_carve]  # take from front of sorted list

        # Find next available index in test folder
        test_start = existing_test + 1
        os.makedirs(test_folder, exist_ok=True)
        for i, img in enumerate(to_move, start=test_start):
            dst_path = os.path.join(test_folder, f"{i}{img.suffix.lower()}")
            shutil.move(str(img), dst_path)

        print(f"  Carved {len(to_move)} test images for '{canonical}' from train.")

        # Update summary
        for row in summary:
            if row[0] == canonical:
                row[1] -= len(to_move)
                row[2] += len(to_move)

    # ==========================================
    # 5. Print summary table
    # ==========================================
    print(f"\n{'Class':<30} {'Train':>6} {'Test':>6} {'Total':>7}")
    print("-" * 52)
    total_train = total_test = 0
    warnings = []
    for canonical, tr, te in summary:
        print(f"  {canonical:<28} {tr:>6} {te:>6} {tr+te:>7}")
        total_train += tr
        total_test += te
        if te < 5:
            warnings.append(f"  WARNING: '{canonical}' has only {te} test images.")
    print("-" * 52)
    print(f"  {'TOTAL':<28} {total_train:>6} {total_test:>6} {total_train+total_test:>7}")

    if warnings:
        print()
        for w in warnings:
            print(w)

    print(f"\nDataset ready at:\n  {DST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", default=DEFAULT_SRC,
        help="Path to the Monuments_SMAI folder (contains train/ and test/ subdirs)"
    )
    args = parser.parse_args()
    main(args.src)
