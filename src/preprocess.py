import os
from PIL import Image

# 1. Define where the data lives
# This trick finds the 'data/processed' folder automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TARGET_SIZE = (224, 224)

# 2. Define valid image types
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def clean_and_resize():
    print(f"🚀 Starting data cleaning in: {DATA_DIR}")
    
    # Loop through all category folders (glass, plastic, biowaste, etc.)
    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        
        # Skip hidden files or files that are not folders
        if not os.path.isdir(category_path):
            continue
            
        print(f"Processing {category}...")
        files = os.listdir(category_path)
        
        count = 0
        for filename in files:
            file_path = os.path.join(category_path, filename)
            
            # Check extension (ignore hidden system files)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue
                
            try:
                # Open image
                img = Image.open(file_path)
                
                # Convert to RGB (Fixed 'P' or 'RGBA' mode issues)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize (Antialias is default in newer Pillow versions)
                img = img.resize(TARGET_SIZE)
                
                # Overwrite the original file with the clean version
                img.save(file_path)
                count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Optional: os.remove(file_path) # Delete corrupt images if you want

        print(f"      ✅ Cleaned {count} images in {category}")

    print("\n🎉 All images are now ready for AI training!")

if __name__ == "__main__":
    clean_and_resize()