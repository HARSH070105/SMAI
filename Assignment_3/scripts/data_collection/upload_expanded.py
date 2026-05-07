"""
Uploads newly downloaded Wikimedia images to Amisha2905/Monuments_SMAI on HuggingFace.

Identifies new images by numeric stem: any file whose stem index is greater than
the original HF train count for that class is "new" (downloaded by expand_dataset.py).

Usage:
  huggingface-cli login        # one-time, or set HF_TOKEN env var
  python scripts/upload_expanded.py
  python scripts/upload_expanded.py --dry-run   # just print what would be uploaded
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "train")
HF_TRAIN   = os.path.join(ROOT_DIR, "..", "Monuments_SMAI_local", "Amisha2905", "Monuments_SMAI", "train")

REPO_ID    = "Amisha2905/Monuments_SMAI"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# canonical class name → original folder name in the HF download
ORIG_FOLDER = {
    "Hawa mahal": "hawa mahal pics",
    "India_gate": "India gate pics",
}


def count_hf_originals(canonical: str) -> int:
    folder_name = ORIG_FOLDER.get(canonical, canonical)
    folder = os.path.join(HF_TRAIN, folder_name)
    if not os.path.isdir(folder):
        return 0
    return sum(1 for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def get_new_images(canonical: str):
    """Return paths of images whose numeric stem > original HF train count."""
    orig_count = count_hf_originals(canonical)
    data_folder = os.path.join(DATA_TRAIN, canonical)
    if not os.path.isdir(data_folder):
        return []
    new = []
    for p in Path(data_folder).iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        if idx > orig_count:
            new.append(p)
    return sorted(new, key=lambda p: int(p.stem))


def main(dry_run: bool):
    api = HfApi()
    operations = []
    total = 0

    for cls_dir in sorted(Path(DATA_TRAIN).iterdir()):
        if not cls_dir.is_dir():
            continue
        canonical = cls_dir.name
        new_imgs = get_new_images(canonical)
        if not new_imgs:
            print(f"  [{canonical}] 0 new — skipping")
            continue
        print(f"  [{canonical}] {len(new_imgs)} new images")
        total += len(new_imgs)
        for img in new_imgs:
            operations.append(CommitOperationAdd(
                path_in_repo=f"train/{canonical}/{img.name}",
                path_or_fileobj=str(img),
            ))

    print(f"\nTotal new images to upload: {total}")

    if dry_run:
        print("Dry run — nothing uploaded.")
        return

    if not operations:
        print("Nothing to upload.")
        return

    print("Uploading as a single PR commit (this may take a few minutes)...")
    result = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {total} expanded training images from Wikimedia Commons",
        create_pr=True,
    )
    print(f"\nDone! PR URL: {result.pr_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without actually uploading")
    args = parser.parse_args()
    main(args.dry_run)
