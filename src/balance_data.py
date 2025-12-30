import os
import random
import glob

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIOWASTE_DIR = os.path.join(BASE_DIR, "data", "processed", "biowaste")

def limit_data(folder_path, limit=500):
    all_files = glob.glob(os.path.join(folder_path, "*"))
    
    # Only run if we have WAY too many
    if len(all_files) <= limit:
        print(f"✅ {os.path.basename(folder_path)} is already safe ({len(all_files)} images).")
        return

    print(f"⚠️  Reducing {os.path.basename(folder_path)} from {len(all_files)} -> {limit} images...")
    
    # Shuffle and pick the ones to DELETE
    random.shuffle(all_files)
    files_to_delete = all_files[limit:] # Keep the first 'limit', delete the rest
    
    for f in files_to_delete:
        os.remove(f)
        
    print(f"🗑️  Deleted {len(files_to_delete)} extra images. Count is now {limit}.")

if __name__ == "__main__":
    # We LIMIT Biowaste to 500 so it stops bullying the other classes
    limit_data(BIOWASTE_DIR, limit=500)