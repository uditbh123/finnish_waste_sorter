import os

# Point this to your processed data folder
data_dir = os.path.join("data", "processed")

print(f"📂 Checking folder: {os.path.abspath(data_dir)}")

if os.path.exists(data_dir):
    folders = os.listdir(data_dir)
    print(f"found folders: {folders}")
    
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            print(f"   - {folder}: {len(image_files)} valid images")
            if len(image_files) == 0:
                print(f"     ⚠️ WARNING: {folder} is empty or has no valid images!")
else:
    print("❌ Error: data/processed folder does not exist.")