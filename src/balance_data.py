import os
import random

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
BIOWASTE_DIR = os.path.join(DATA_DIR, "biowaste")

# 2. Target Count
TARGET_COUNT = 1500

def balance_biowaste():
    # Check if folder exists
    if not os.path.exists(BIOWASTE_DIR):
        print(f"Error: Could not find folder at {BIOWASTE_DIR}")
        return
    
    # GET list of all files
    files = os.listdir(BIOWASTE_DIR)
    current_count = len(files)

    print(f"Current Biowaste images: {current_count}")

    # Check if we need to delete anything
    if current_count <= TARGET_COUNT:
        print("Count is already low enough. No action needed.")
        return
    
    # 3. The culling (deletion logic)
    to_remove_count = current_count - TARGET_COUNT
    print(f"Removing {to_remove_count} random images to balance the dataset...")

    # Shuffle list
    random.shuffle(files)

    # Select victims
    files_to_delete = files[:to_remove_count]

    # Delete them
    deleted_count = 0
    for filename in files_to_delete:
        file_path = os.path.join(BIOWASTE_DIR, filename)
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

    print(f"Successfully deleted {deleted_count} images.")
    print(f"Biowaste folder now has {TARGET_COUNT} images.")

# CORRECT: This is now all the way to the left (Global Scope)
if __name__ == "__main__":
    balance_biowaste()