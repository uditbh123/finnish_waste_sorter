import os
import time
import random
import requests
from duckduckgo_search import DDGS

# 1. Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# 2. Search Terms
SEARCHES = {
    "plastic": ["used plastic bottle", "crushed yoghurt cup", "plastic food packaging waste", "shampoo bottle trash"],
    "glass": ["empty jam jar waste", "glass bottle trash", "broken glass jar"],
    "metal": ["crushed aluminum can", "tin food can waste", "metal lid trash"],
    "cardboard": ["brown cardboard box waste", "egg carton waste", "milk carton trash"]
}

MAX_IMAGES_PER_TERM = 20 

def download_from_search():
    print("🚀 Starting Data Injection (Stealth Mode)...")

    for class_name, queries in SEARCHES.items():
        save_folder = os.path.join(DATA_DIR, class_name)
        os.makedirs(save_folder, exist_ok=True)
        
        print(f"\n📦 Processing Class: {class_name.upper()}")
        
        for query in queries:
            print(f"   🔎 Querying: '{query}'...")
            
            # 🟢 FIX: Initialize DDGS inside the loop for a fresh session every time
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(
                        query, 
                        region="wt-wt", 
                        safesearch="off", 
                        max_results=MAX_IMAGES_PER_TERM
                    ))
            except Exception as e:
                print(f"      ⚠️ Search failed for '{query}': {e}")
                time.sleep(5) # Cooldown on error
                continue

            count = 0
            for r in results:
                image_url = r.get('image')
                if not image_url: continue

                try:
                    # Download with a timeout
                    response = requests.get(image_url, timeout=4)
                    if response.status_code == 200:
                        timestamp = int(time.time() * 1000)
                        filename = f"web_{class_name}_{timestamp}_{count}.jpg"
                        filepath = os.path.join(save_folder, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        count += 1
                except Exception:
                    pass # Skip bad links

            print(f"      ✅ Downloaded {count} images")
            
            # 🟢 FIX: Random sleep to look like a human
            sleep_time = random.uniform(3, 6)
            print(f"      💤 Sleeping for {sleep_time:.1f}s...")
            time.sleep(sleep_time)

if __name__ == "__main__":
    download_from_search()
    print("\n🎉 Data Injection Complete! Now check the folders.")