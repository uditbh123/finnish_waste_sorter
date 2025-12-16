# ♻️ SortWise: Finnish Waste Classification AI

**An AI-powered computer vision system designed to help residents in Finland sort household waste correctly according to HSY/local guidelines.**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Status](https://img.shields.io/badge/Status-Prototype-green)

## 🇫🇮 The Problem
New residents and even locals in Finland often struggle with complex recycling rules (e.g., distinguishing *Kartonki* form *Paperi*, or knowing what counts as *Biojäte*). Incorrect sorting contaminates recycling batches.

## 💡 The Solution
SortWise uses a **MobileNetV2** Convolutional Neural Network (Transfer Learning) to classify waste images into 6 categories standard in Finnish waste management:
* **Biowaste (Biojäte)**
* **Plastic (Muovi)**
* **Cardboard (Kartonki)**
* **Glass (Lasi)**
* **Metal (Metalli)**
* **Mixed Waste (Sekajäte)**

## 🛠️ Technical Implementation
* **Data Pipeline:** Custom ETL scripts (`src/preprocess.py`) to standardize input images to 224x224 RGB.
* **Handling Imbalance:** Implemented automated undersampling (`src/balance_data.py`) to manage class disparity (e.g., balancing 13k Bio images vs 400 Metal images).
* **Model:** Fine-tuned MobileNetV2 with custom head (GlobalAveragePooling + Dropout) for efficiency on consumer hardware.
* **Data Augmentation:** Used `ImageDataGenerator` (rotation, zoom, shear) to improve generalization on small classes like Mixed Waste.


## 📅 Dev Log: December 16, 2025
*Today marked a major milestone: moving from data processing to a fully trained, inference-ready AI model.*

### 🚧 Challenges Faced
We encountered significant environment issues involving Python 3.14 compatibility with TensorFlow. We resolved this by building a dedicated Anaconda environment (`waste_sorter`) running Python 3.10, ensuring a stable training pipeline.

### 🧪 Real-World Testing & Insights
After training the MobileNetV2 model (achieving **88% training accuracy** and **80% validation accuracy**), we conducted three specific "stress tests" on real-world images to evaluate the model's logic:

1.  **The "Valio Milk" Test (Fail):**
    * *Input:* A red Valio milk carton with a picture of a cow.
    * *Prediction:* **Biowaste** (43% confidence).
    * *Insight:* The model likely associated the cow (animal/organic) and the color red with biological waste. This highlights the need for **Finnish-specific fine-tuning** (local branding awareness).

2.  **The "Intact Coke Bottle" Test (Fail):**
    * *Input:* A clear, smooth Coca-Cola bottle.
    * *Prediction:* **Glass** (79% confidence).
    * *Insight:* The model struggled to distinguish clear plastic from glass based on transparency and specularity alone.

3.  **The "Crushed Bottle" Test (SUCCESS ✅):**
    * *Input:* A crushed, crinkled water bottle.
    * *Prediction:* **Plastic** (73.8% confidence).
    * *Insight:* **Success!** The model correctly identified plastic when the object was deformed. Since glass shatters and does not crinkle, the AI successfully used the object's physical properties/geometry to make the correct classification.

---

## 📂 Project Structure

```text
finnish-waste-sorter/
├── data/               # Raw and Processed data (GitIgnored)
├── models/             # Trained .h5 models
├── src/                # Source code
│   ├── preprocess.py   # Image resizing and cleaning
│   ├── balance_data.py # Class balancing logic
│   ├── train_model.py  # MobileNetV2 training loop
│   └── predict.py      # Inference script for testing real images
├── .gitignore          # Files to exclude from Git
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation