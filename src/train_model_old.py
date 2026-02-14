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

# 🟢 NEW: Define Augmentation completely separate from the model
# This prevents the "Pickle" crash because the model doesn't need to save these layers.
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.3),
    RandomZoom(0.3),
    RandomContrast(0.3),
    RandomBrightness(0.3, value_range=(0, 255)) 
])

def build_model(num_classes):
    """
    Builds a pure MobileNetV2 model (No augmentation layers inside).
    """
    inputs = Input(shape=(224, 224, 3))
    
    # Base Model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Dropout / Dense Head
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu', name='dense_intermediate')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return base_model, model

def get_callbacks(filename):
    return [
        EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True, # We can safely use True now!
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
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

    # 1. Load Data
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

    # 🟢 NEW: Apply augmentation to the DATASET, not the MODEL
    # The lambda function ensures augmentation only happens during training
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optimize pipeline
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Build Model
    base_model, model = build_model(len(class_names))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- PHASE 1: HEAD ---
    print(f"\n🧠 Phase 1: Training Head ({EPOCHS_HEAD} epochs)...")
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks("phase1_head.keras"),
        verbose=1
    )

    # --- PHASE 2: FINE TUNE ---
    print(f"\n🔓 Phase 2: Unfreezing top layers ({EPOCHS_FINE} epochs)...")
    
    # Reload best weights from Phase 1 to be safe
    model.load_weights(os.path.join(MODEL_DIR, "phase1_head.keras"))
    
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks("phase2_finetuned.keras"), 
        verbose=1
    )

    print("\n🎉 TRAINING COMPLETE!")
    print(f"💾 Final Model: {os.path.join(MODEL_DIR, 'phase2_finetuned.keras')}")

if __name__ == "__main__":
    main()