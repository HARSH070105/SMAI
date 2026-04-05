import os

# 🔹 CHANGE THIS PATH
ROOT_DIR = r"C:\Users\harsh\OneDrive\Desktop\SMAI\Assignment_2\Dataset"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def rename_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]

    files.sort()
    n = len(files)

    if n == 0:
        return 0

    pad = len(str(n))  # dynamic padding

    # 🔹 Pass 1: temp rename (avoid overwrite)
    temp_names = []
    for i, f in enumerate(files):
        old = os.path.join(folder_path, f)
        ext = os.path.splitext(f)[1]
        temp = os.path.join(folder_path, f"__tmp__{i}{ext}")
        os.rename(old, temp)
        temp_names.append(temp)

    # 🔹 Pass 2: final rename
    for i, temp in enumerate(temp_names, start=1):
        ext = os.path.splitext(temp)[1]
        new = os.path.join(folder_path, f"{str(i).zfill(pad)}{ext}")
        os.rename(temp, new)

    return n


def main():
    for folder in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, folder)

        if os.path.isdir(folder_path):
            count = rename_in_folder(folder_path)
            print(f"{folder}: {count} images renamed")


if __name__ == "__main__":
    main()