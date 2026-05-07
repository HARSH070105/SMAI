"""
Downloads images from Wikimedia Commons for NEW monument classes
(beyond the original 24) and adds them to the dataset.

Creates both train and test splits for each new class.
Run from repo root: python scripts/add_new_classes.py

Usage:
  python scripts/add_new_classes.py
  python scripts/add_new_classes.py --target 120 --test-size 15 --dry-run
"""

import os
import time
import argparse
import requests
from pathlib import Path

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "train")
TEST_DIR  = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "test")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS     = {"User-Agent": "SMAI-A3-MonumentApp/1.0 (academic project)"}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}

# ==========================================
# New classes: folder name → Wikimedia search query
# ==========================================

NEW_CLASSES = {
    # T12.2 — Forts of Maharashtra
    "Raigad Fort":              "Raigad Fort Maharashtra",
    "Sinhagad Fort":            "Sinhagad Fort Pune",
    "Pratapgad Fort":           "Pratapgad Fort",
    "Shivneri Fort":            "Shivneri Fort",
    "Lohagad Fort":             "Lohagad Fort",
    "Panhala Fort":             "Panhala Fort",
    "Sindhudurg Fort":          "Sindhudurg Fort",

    # T12.3 — Temples of Tamil Nadu (tanjavur temple already in 24)
    "Meenakshi Temple":         "Meenakshi Amman Temple Madurai",
    "Shore Temple":             "Shore Temple Mahabalipuram",
    "Ramanathaswamy Temple":    "Ramanathaswamy Temple Rameswaram",
    "Kapaleeshwarar Temple":    "Kapaleeshwarar Temple Chennai",
    "Ranganathaswamy Temple":   "Ranganathaswamy Temple Srirangam",
    "Nataraja Temple":          "Thillai Nataraja Temple Chidambaram",

    # T12.4 — Mughal architecture (tajmahal, Humayun_s Tomb, Fatehpur Sikri,
    #          alai_darwaza, alai_minar, qutub_minar, jamali_kamali_tomb already in 24)
    "Agra Fort":                "Agra Fort",
    "Red Fort Delhi":           "Red Fort Delhi",
    "Jama Masjid Delhi":        "Jama Masjid Delhi",
    "Itmad-ud-Daulah":          "Itmad-ud-Daulah Agra",
    "Bibi Ka Maqbara":          "Bibi Ka Maqbara Aurangabad",
    "Safdarjung Tomb":          "Safdarjung Tomb Delhi",
    "Akbar Tomb Sikandra":      "Tomb of Akbar Sikandra",

    # T12.5 — Hampi monuments
    "Virupaksha Temple Hampi":  "Virupaksha Temple Hampi",
    "Vittala Temple Hampi":     "Vittala Temple Hampi",
    "Lotus Mahal Hampi":        "Lotus Mahal Hampi",

    # T12.6 — Stepwells of India
    "Rani ki Vav":              "Rani ki Vav Patan",
    "Chand Baori":              "Chand Baori Abhaneri",
    "Adalaj Stepwell":          "Adalaj Stepwell Gujarat",
}


# ==========================================
# Helpers (same as expand_dataset.py)
# ==========================================

def search_and_get_urls(query: str, limit: int = 50, offset: int = 0):
    params = {
        "action":       "query",
        "generator":    "search",
        "gsrnamespace": "6",
        "gsrsearch":    query,
        "gsrlimit":     limit,
        "gsroffset":    offset,
        "prop":         "imageinfo",
        "iiprop":       "url|size|mime|thumburl",
        "iiurlwidth":   800,
        "format":       "json",
    }
    try:
        r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=20)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        results = []
        for page in pages.values():
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info     = infos[0]
            mime     = info.get("mime", "")
            thumburl = info.get("thumburl", "")
            width    = info.get("width", 0)
            height   = info.get("height", 0)
            if mime in ("image/jpeg", "image/png") and width >= 300 and height >= 300 and thumburl:
                ext = ".jpg" if mime == "image/jpeg" else ".png"
                results.append((thumburl, ext))
        next_offset = data.get("continue", {}).get("gsroffset")
        return results, next_offset
    except Exception:
        return [], None


def download_image(url: str, dst_path: str, max_bytes: int = 4_000_000) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=8, stream=True)
        if r.status_code == 429:
            time.sleep(60)
            return False
        if r.status_code != 200:
            return False
        size = 0
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                size += len(chunk)
                if size > max_bytes:
                    os.remove(dst_path)
                    return False
                f.write(chunk)
        if os.path.getsize(dst_path) < 5000:
            os.remove(dst_path)
            return False
        return True
    except Exception:
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False


# ==========================================
# Per-class download
# ==========================================

def add_class(canonical: str, query: str, target: int, test_size: int, per_request: int):
    train_folder = os.path.join(TRAIN_DIR, canonical)
    test_folder  = os.path.join(TEST_DIR,  canonical)

    # Skip if already has enough images
    existing_train = sum(1 for p in Path(train_folder).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(train_folder) else 0
    existing_test  = sum(1 for p in Path(test_folder).iterdir()  if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(test_folder)  else 0
    existing_total = existing_train + existing_test

    if existing_total >= target:
        print(f"  [{canonical}] {existing_total} images already — skipping.")
        return 0

    needed = target - existing_total
    print(f"  [{canonical}] downloading up to {needed} images (target={target}, test_size={test_size})...")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder,  exist_ok=True)

    downloaded = 0
    offset     = 0
    # Index counters
    train_idx = existing_train + 1
    test_idx  = existing_test  + 1

    while downloaded < needed:
        results, offset = search_and_get_urls(query, limit=per_request, offset=offset)
        if not results:
            break

        for url, ext in results:
            if downloaded >= needed:
                break

            # First test_size images go to test, rest to train
            total_so_far = existing_total + downloaded
            if total_so_far < test_size:
                dst_path = os.path.join(test_folder, f"{test_idx}{ext}")
                if download_image(url, dst_path):
                    downloaded += 1
                    test_idx   += 1
                    print(f"    {downloaded}/{needed} → test", flush=True)
            else:
                dst_path = os.path.join(train_folder, f"{train_idx}{ext}")
                if download_image(url, dst_path):
                    downloaded += 1
                    train_idx  += 1
                    print(f"    {downloaded}/{needed} → train", flush=True)

            time.sleep(32)

        if offset is None:
            break

    print(f"    → Downloaded {downloaded} images.")
    return downloaded


# ==========================================
# Main
# ==========================================

def main(target: int, test_size: int, per_request: int, dry_run: bool):
    print(f"Target per class : {target}  (test split: {test_size})")
    print(f"New classes      : {len(NEW_CLASSES)}")
    print()

    if dry_run:
        for canonical, query in NEW_CLASSES.items():
            train_folder = os.path.join(TRAIN_DIR, canonical)
            test_folder  = os.path.join(TEST_DIR,  canonical)
            tr = sum(1 for p in Path(train_folder).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(train_folder) else 0
            te = sum(1 for p in Path(test_folder).iterdir()  if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(test_folder)  else 0
            print(f"  {canonical:<30} existing: train={tr} test={te}  query: {query}")
        return

    total_downloaded = 0
    for canonical, query in NEW_CLASSES.items():
        n = add_class(canonical, query, target, test_size, per_request)
        total_downloaded += n

    print(f"\nDone. Total new images downloaded: {total_downloaded}")
    print(f"\n{'Class':<35} {'Train':>6} {'Test':>6} {'Total':>7}")
    print("-" * 57)
    for canonical in NEW_CLASSES:
        tr = sum(1 for p in Path(os.path.join(TRAIN_DIR, canonical)).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(os.path.join(TRAIN_DIR, canonical)) else 0
        te = sum(1 for p in Path(os.path.join(TEST_DIR,  canonical)).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(os.path.join(TEST_DIR,  canonical)) else 0
        print(f"  {canonical:<33} {tr:>6} {te:>6} {tr+te:>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int, default=120,
                        help="Target total images per new class (default: 120)")
    parser.add_argument("--test-size",   type=int, default=15,
                        help="Images to put in test split per class (default: 15)")
    parser.add_argument("--per-request", type=int, default=50,
                        help="Files per Wikimedia API call (default: 50)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print class list without downloading")
    args = parser.parse_args()
    main(args.target, args.test_size, args.per_request, args.dry_run)
