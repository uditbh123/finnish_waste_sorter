import os 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_sorter_best.keras")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    print(f" Loading Model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model Loaded Successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    print("\n Loading Validation Dataset...")
    # 🟢 CRITICAL: shuffle=False so the predictions match the true labels in order!
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, 
        validation_split=0.2,
        subset="validation",
        seed = 123,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        label_mode = 'categorical',
        shuffle = True
    )

    class_names = val_ds.class_names
    # Apply Normalization(matches training)
    normalization = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    val_ds_normalized = val_ds.map(lambda x, y: (normalization(x), y))

    print("\n🤖 Running predictions on validation data. This might take a minute...")
    
    y_true = []
    predictions = []

    # 🟢 CRITICAL FIX: Extract true labels AND make predictions in the exact same loop!
    # This guarantees they never get out of sync, even if the dataset shuffles.
    for images, labels in val_ds_normalized:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        
        # Predict on this exact batch of images immediately
        batch_preds = model.predict(images, verbose=0)
        predictions.extend(batch_preds)
        
    y_true = np.array(y_true)
    y_pred = np.argmax(predictions, axis=1)

    # 3. Generate Confusion Matrix

    # 3. Generate Confusion Matrix
    print("\n Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    # 4. draw the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('SortWise Confusion Matrix (Validation Data)', fontsize=16)
    plt.ylabel('Actual Trash Class', fontsize=12)
    plt.xlabel('Predicted Trash Class', fontsize=12)
    
    # 5. Save the image
    save_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Graphic saved successfully to: {save_path}")

    # 6. Print Text Report
    print("\n📝 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()