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

## 📅 Dev Log: December 30, 2025
*Milestone: Phase 2 Fine-Tuning Complete & Batch Testing.*

### 🚀 Progress
We successfully fine-tuned the top layers of the model, achieving a **Training Accuracy of 95.3%** and **Validation Accuracy of ~81.6%**. The model is now capable of running batch predictions on folders of mixed images.

### 🧪 Latest Stress Test Results
We ran a "Blind Batch Test" on random internet images. The results revealed clear strengths and biases:

#### ✅ The Wins
1.  **Biowaste Mastery:** The model is incredibly confident (>99%) with organic matter like banana peels, compost, and vegetables.
2.  **Texture Recognition:** Successfully identified an **IKEA Box** as *Cardboard* (86%) despite the complex logos, proving it is learning feature shapes.

#### ⚠️ The "Background Bias" Discovery
The model revealed a critical flaw in how it perceives context:
* **The "Dirty Bottle" Error:**
    * *Input:* A plastic bottle sitting on a pile of dirt/trash.
    * *Prediction:* **Biowaste** (99% confidence).
    * *Insight:* The model ignored the bottle and classified the **background** (dirt) as compost. This suggests the training data for "Plastic" was too clean, while "Biowaste" data was mostly messy/brown.

* **The "Color Trap":**
    * *Input:* A plain brown cardboard box.
    * *Prediction:* **Biowaste** (62% confidence).
    * *Insight:* Without distinct texture cues (like corrugation), the model confuses the color **brown** with potato peels or leaves.

### 🔮 Next Steps
* **Data Augmentation:** Implement rotation and zooming to force the model to focus on object shape rather than background color.
* **Targeted Data Collection:** Add more images of "Plastic on dirt" and "Clean Cardboard" to break the current biases.

## 📅 Dev Log: January 6, 2026
*Milestone: Visual Interface & Critical Logic Fixes.*

### 🚀 Major Feature: Web Interface (GUI)
Moved away from command-line scripts and built a **Streamlit Web App** (`src/app.py`).
* **Why:** To visualize model confidence scores and debug "borderline" predictions in real-time.
* **Feature:** Users can now drag-and-drop images and see a breakdown of probabilities (e.g., "55% Plastic, 45% Paper").

### 🐛 Bug Fix: The "Plastic is Paper" Error
**The Problem:** The model was consistently labeling plastic bottles as "Paper" in the CLI, despite high confidence.
**The Root Cause:** A "silent shift" in class mapping.
* The training data contained **5 classes** (Paper folder was deleted previously).
* The prediction code expected **6 classes** (including Paper).
* **Result:** The model predicted Index 4 (Plastic), but the code mapped Index 4 to "Paper".
**The Fix:** Updated `CLASS_NAMES` in `app.py` to dynamically match the actual 5-class training structure.

### 📉 Current Challenge: The "Brown Box" Bias
**Issue:** The model frequently misclassifies **Cardboard** as **Biowaste** (~60% confidence).
**Analysis:** The model is over-relying on **color** (Brown) rather than **shape** (Square edges). Since most biowaste images are brownish (dirt/compost), the model assumes "Brown = Bio".

### 🔮 Next Steps
* **Advanced Augmentation:** Investigate "Color Jitter" or Edge Detection to force the model to focus on geometry over color.
* **UI Polish:** Add a "Feedback Loop" button so users can correct the AI when it makes a mistake.
---



## 📂 Project Structure

```text
finnish-waste-sorter/
├── app/                # Application specific resources
├── data/               # Raw and Processed data (GitIgnored)
├── models/             # Trained .h5 models
├── src/                # Source code
│   ├── app.py          # Streamlit Web Interface (Main Entry Point)
│   ├── augment_finnish.py # Specific augmentation for Finnish brands
│   ├── balance_data.py # Class balancing logic
│   ├── check_data.py   # Utility to verify dataset integrity
│   ├── predict.py      # CLI Batch inference script
│   ├── preprocess.py   # Image resizing and cleaning
│   ├── reset_data.py   # Utility to reset processed data
│   └── train_model.py  # MobileNetV2 training loop
├── test_dump/          # Local testing images (GitIgnored)
├── .gitignore          # Files to exclude from Git
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation