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

## 📂 Project Structure
finnish-waste-sorter/
├── data/               # Raw and Processed data (GitIgnored)
├── models/             # Trained .h5 models
├── src/                # Source code
│   ├── preprocess.py   # Image resizing and cleaning
│   ├── balance_data.py # Class balancing logic
│   └── train_model.py  # MobileNetV2 training loop
├── .gitignore          # Files to exclude from Git
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation