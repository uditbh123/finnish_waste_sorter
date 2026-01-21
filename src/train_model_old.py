import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 20
EPOCHS_FINE = 30
LEARNING_RATE = 0.0001

def build_augmented_model(num_classes):
    inputs = Input(shape=(224, 224, 3))

    # 🟢 AGGRESSIVE AUGMENTATION (To fight Color Bias)
    x = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.3),
        RandomZoom(0.3),
        RandomContrast(0.3),
        # Crucial for Finnish winter lighting vs indoor lighting
        RandomBrightness(0.3, value_range=(0, 255)) 
    ], name="augmentation_layer")(inputs)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 🟢 HIGH DROPOUT (To fight Overfitting)
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu', name='dense_intermediate')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return base_model, model

def get_callbacks(filename):
    return [
        EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        # 🟢 FIX: Separate filenames to prevent overwriting
        ModelCheckpoint(
            os.path.join(MODEL_DIR, filename),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

def main():
    print(f"🚀 Found GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"⏳ Loading dataset from: {DATA_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )

    class_names = train_ds.class_names
    print(f"✅ Classes found: {class_names}")

    # ⚠️ IF YOU GET MEMORY ERRORS, REMOVE .cache()
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    base_model, model = build_augmented_model(len(class_names))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- PHASE 1: HEAD ---
    print(f"\n🧠 Phase 1: Training Head ({EPOCHS_HEAD} epochs)...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
        callbacks=get_callbacks("phase1_head.keras"), # Save as .keras (modern format)
        verbose=1
    )

    # --- PHASE 2: FINE TUNE ---
    print(f"\n🔓 Phase 2: Unfreezing top layers ({EPOCHS_FINE} epochs)...")
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10), # Lower LR for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
        callbacks=get_callbacks("phase2_finetuned.keras"), # Save final model separately
        verbose=1
    )

    print("\n🎉 TRAINING COMPLETE!")
    print(f"💾 Final Model: {os.path.join(MODEL_DIR, 'phase2_finetuned.keras')}")

if __name__ == "__main__":
    main()