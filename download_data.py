import os
import zipfile
import subprocess
import sys

DATASET_SLUG = "sachinkumar413/alzheimer-mri-dataset"
ZIP_NAME = "alzheimer-mri-dataset.zip"
EXTRACT_DIR = "dataset"


def check_kaggle_token():
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    token_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(token_path):
        print("ERROR: kaggle.json not found at", token_path)
        print("\nSetup steps:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Scroll to 'API' section and click 'Create New Token'")
        print("  3. Move the downloaded kaggle.json to:", kaggle_dir)
        if sys.platform == "win32":
            print("  4. On Windows, run: icacls %USERPROFILE%\\.kaggle\\kaggle.json /inheritance:r /grant:r \"%USERNAME%\":R")
        else:
            print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("Kaggle token found.")


def download_dataset():
    print(f"Downloading dataset: {DATASET_SLUG} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Download failed:\n", result.stderr)
        sys.exit(1)
    print(result.stdout)
    print("Download complete.")


def unzip_dataset():
    if not os.path.exists(ZIP_NAME):
        print(f"ERROR: {ZIP_NAME} not found. Run download first.")
        sys.exit(1)

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print(f"Extracting {ZIP_NAME} -> {EXTRACT_DIR}/ ...")
    with zipfile.ZipFile(ZIP_NAME, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("Extraction complete.")
    print("\nDataset folder structure:")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        depth = root.replace(EXTRACT_DIR, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth >= 2:
            break


def verify_classes():
    expected = {"NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"}
    found = set()
    for root, dirs, _ in os.walk(EXTRACT_DIR):
        for d in dirs:
            if d in expected:
                found.add(d)
    if found == expected:
        print("\nAll 4 class folders found:", sorted(found))
    else:
        missing = expected - found
        print("\nWARNING: Missing class folders:", missing)
        print("Found:", found)
        print("Check the extracted folder structure manually.")


if __name__ == "__main__":
    check_kaggle_token()
    download_dataset()
    unzip_dataset()
    verify_classes()
    print("\nDone. Run: python train.py")
