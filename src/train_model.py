import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 8      # Phase 1: Training the new layers
EPOCHS_FINE = 8      # Phase 2: Fine-tuning
LEARNING_RATE = 0.0001

def build_augmented_model(num_classes):
    """
    Builds a MobileNetV2 with built-in Data Augmentation layers.
    This forces the AI to learn shapes, not just backgrounds.
    """
    # 1. Define Input
    inputs = Input(shape=(224, 224, 3))

    # 2. Data Augmentation Layers (The "Brain Surgery")
    x = Sequential([
        RandomFlip("horizontal_and_vertical"), # Flip left/right/up/down
        RandomRotation(0.2),                   # Rotate up to 20%
        RandomZoom(0.2),                       # Zoom in/out up to 20%
        RandomContrast(0.2),                   # Adjust contrast (lighting changes)
    ], name="augmentation_layer")(inputs)

    # 3. Base Model (MobileNetV2)
    # I pass the augmented images 'x' into MobileNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False  # Freeze base model initially

    # 4. Custom Head (The classifier)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Prevents overfitting
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return base_model, model

def main():
    print(f"🚀 Found GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # 1. Load Data
    print(f"⏳ Loading dataset from: {DATA_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    # --- FIX: Get class names BEFORE caching ---
    class_names = train_ds.class_names
    print(f"✅ Classes found: {class_names}")
    # -------------------------------------------

    # Optimize for performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Build Model
    base_model, model = build_augmented_model(len(class_names))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. Phase 1: Train Head
    print("\n🧠 Phase 1: Training Head (with Augmentation)...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # 4. Phase 2: Fine-Tuning
    print("\n🔓 Phase 2: Unfreezing top layers...")
    base_model.trainable = True

    # Freeze the bottom layers (MobileNetV2 has 155 layers)
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10), # Lower LR for fine-tuning
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

    # 5. Save
    save_path = os.path.join(MODEL_DIR, "waste_sorter_augmented.h5")
    model.save(save_path)
    print(f"🎉 Augmented Model Saved to: {save_path}")

if __name__ == "__main__":
    main()