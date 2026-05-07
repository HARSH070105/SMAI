"""
Downloads images from Wikimedia Commons for Vedant's 13 new monument classes.
Run from repo root: python scripts/add_classes_vedant.py

Usage:
  python scripts/add_classes_vedant.py
  python scripts/add_classes_vedant.py --dry-run
"""

import os
import time
import random
import argparse
import requests
from pathlib import Path

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "train")
TEST_DIR  = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "test")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS     = {"User-Agent": "SMAI-A3-MonumentApp/1.0 (academic project)"}
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}

MY_CLASSES = {
    # T12.4 — Mughal architecture
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

    # T12.6 — Stepwells
    "Rani ki Vav":              "Rani ki Vav Patan",
    "Chand Baori":              "Chand Baori Abhaneri",
    "Adalaj Stepwell":          "Adalaj Stepwell Gujarat",
}


def search_and_get_urls(query, limit=50, offset=0):
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


def download_image(url, dst_path, max_bytes=4_000_000):
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


def add_class(canonical, query, target, test_size, per_request):
    train_folder = os.path.join(TRAIN_DIR, canonical)
    test_folder  = os.path.join(TEST_DIR,  canonical)

    existing_train = sum(1 for p in Path(train_folder).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(train_folder) else 0
    existing_test  = sum(1 for p in Path(test_folder).iterdir()  if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(test_folder)  else 0
    existing_total = existing_train + existing_test

    if existing_total >= target:
        print(f"  [{canonical}] {existing_total} images already — skipping.")
        return 0

    needed = target - existing_total
    print(f"  [{canonical}] downloading up to {needed} images to train, then splitting...")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder,  exist_ok=True)

    downloaded = 0
    offset     = 0
    train_idx  = existing_train + 1

    # Download everything to train first
    while downloaded < needed:
        results, offset = search_and_get_urls(query, limit=per_request, offset=offset)
        if not results:
            break

        for url, ext in results:
            if downloaded >= needed:
                break
            dst_path = os.path.join(train_folder, f"{train_idx}{ext}")
            if download_image(url, dst_path):
                downloaded += 1
                train_idx  += 1
                print(f"    {downloaded}/{needed} downloaded", flush=True)
            time.sleep(32)

        if offset is None:
            break

    # Randomly move test_size images from train to test
    all_train = [p for p in Path(train_folder).iterdir() if p.suffix.lower() in IMAGE_EXTS]
    existing_test_count = sum(1 for p in Path(test_folder).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(test_folder) else 0
    still_needed_test = max(0, test_size - existing_test_count)
    to_move = random.sample(all_train, min(still_needed_test, len(all_train)))
    for i, img in enumerate(to_move, start=existing_test_count + 1):
        dst = os.path.join(test_folder, f"{i}{img.suffix}")
        os.rename(str(img), dst)
    print(f"    → Downloaded {downloaded}, moved {len(to_move)} to test randomly.")
    return downloaded


def main(target, test_size, per_request, dry_run):
    print(f"Target per class : {target}  (test split: {test_size})")
    print(f"Classes          : {len(MY_CLASSES)}")
    print()

    if dry_run:
        for canonical, query in MY_CLASSES.items():
            print(f"  {canonical:<35} query: {query}")
        return

    total = 0
    for canonical, query in MY_CLASSES.items():
        total += add_class(canonical, query, target, test_size, per_request)

    print(f"\nDone. Total downloaded: {total}")
    print(f"\n{'Class':<35} {'Train':>6} {'Test':>6} {'Total':>7}")
    print("-" * 57)
    for canonical in MY_CLASSES:
        tr = sum(1 for p in Path(os.path.join(TRAIN_DIR, canonical)).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(os.path.join(TRAIN_DIR, canonical)) else 0
        te = sum(1 for p in Path(os.path.join(TEST_DIR,  canonical)).iterdir() if p.suffix.lower() in IMAGE_EXTS) if os.path.isdir(os.path.join(TEST_DIR,  canonical)) else 0
        print(f"  {canonical:<33} {tr:>6} {te:>6} {tr+te:>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int, default=120)
    parser.add_argument("--test-size",   type=int, default=15)
    parser.add_argument("--per-request", type=int, default=50)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()
    main(args.target, args.test_size, args.per_request, args.dry_run)
