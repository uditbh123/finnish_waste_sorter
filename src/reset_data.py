import os
import glob

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def delete_generated_files(folder_name):
    # This pattern matches ONLY files that contain "_aug_"
    # It will NOT touch valio1.jpg or cardboard1.jpg
    search_pattern = os.path.join(PROCESSED_DIR, folder_name, "*_aug_*")
    files_to_delete = glob.glob(search_pattern)
    
    print(f"\n🗑️  Cleaning '{folder_name}'...")
    if len(files_to_delete) == 0:
        print("   (Clean. No generated files found.)")
        return

    count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            count += 1
        except Exception as e:
            print(f"   Error deleting {file_path}: {e}")
            
    print(f"   ✅ Deleted {count} generated duplicates.")

if __name__ == "__main__":
    print("🚨 STARTING DATA RESET (Deleting generated variants)...")
    
    delete_generated_files("cardboard")
    delete_generated_files("plastic")
    delete_generated_files("biowaste")
    
    print("\n✨ Done. Folders are clean. You have only original images now.")