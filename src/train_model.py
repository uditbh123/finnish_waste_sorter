import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Settings - UPDATED FOR ANTI-OVERFITTING
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 20      # INCREASED: Was 8, now 20
EPOCHS_FINE = 30      # INCREASED: Was 8, now 30
LEARNING_RATE = 0.0001

# 🔥 NEW: Add ColorJitter-style augmentation to built-in layers
def build_augmented_model(num_classes):
    """
    Builds a MobileNetV2 with AGGRESSIVE built-in Data Augmentation.
    🆕 Now includes better augmentation to break color bias.
    """
    # 1. Define Input
    inputs = Input(shape=(224, 224, 3))

    # 2. Data Augmentation Layers (ENHANCED)
    x = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.3),  # INCREASED from 0.2 to 0.3 (more rotation)
        RandomZoom(0.3),      # INCREASED from 0.2 to 0.3
        RandomContrast(0.3),  # INCREASED from 0.2 to 0.3
        
        # 🆕 ADD brightness variation (helps with color bias)
        tf.keras.layers.RandomBrightness(0.3, value_range=(0, 255)),
        
    ], name="augmentation_layer")(inputs)

    # 3. Base Model (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False  # Freeze initially

    # 4. Custom Head with STRONGER regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 🔥 INCREASED Dropout from 0.2 to 0.5
    x = Dropout(0.5)(x)  # Was 0.2, now 0.5 (much stronger)
    
    # 🆕 ADD second dense layer for better feature learning
    x = Dense(256, activation='relu', name='dense_intermediate')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return base_model, model

# 🆕 NEW: Callbacks for smart training
def get_callbacks(model_save_path):
    """
    Prevents overfitting by monitoring validation loss
    """
    return [
        # Stop if validation doesn't improve for 8 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=False,  # Changed to False (ModelCheckpoint handles saving)
            verbose=1
        ),
        
        # Reduce learning rate if stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Save best model automatically
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

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

    # Get class names BEFORE caching
    class_names = train_ds.class_names
    print(f"✅ Classes found: {class_names}")

    # Optimize for performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Build Model
    base_model, model = build_augmented_model(len(class_names))

    # 🆕 COMPILE - Simple version without label_smoothing
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 🆕 Show model summary
    print("\n📊 Model Architecture:")
    model.summary()

    # 3. Phase 1: Train Head (LONGER)
    save_path = os.path.join(MODEL_DIR, "waste_sorter_best.h5")
    
    print(f"\n🧠 Phase 1: Training Head for {EPOCHS_HEAD} epochs...")
    print("   (Will auto-stop early if overfitting detected)")
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(save_path),
        verbose=1
    )

    # 🆕 REPORT Phase 1 results
    print("\n" + "="*60)
    print("📊 PHASE 1 RESULTS:")
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc
    print(f"   Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"   Val Accuracy:   {final_val_acc*100:.2f}%")
    print(f"   Gap:            {gap*100:.2f}%")
    if gap > 0.1:
        print("   ⚠️  Still overfitting (gap > 10%)")
    else:
        print("   ✅ Overfitting under control!")
    print("="*60)

    # 4. Phase 2: Fine-Tuning
    print(f"\n🔓 Phase 2: Unfreezing top layers for {EPOCHS_FINE} epochs...")
    base_model.trainable = True

    # Freeze bottom 100 layers
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"   Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks(save_path),
        verbose=1
    )

    # 5. Final Report
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    
    final_train_acc = history_fine.history['accuracy'][-1]
    final_val_acc = history_fine.history['val_accuracy'][-1]
    final_gap = final_train_acc - final_val_acc
    
    print(f"📊 FINAL RESULTS:")
    print(f"   Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"   Val Accuracy:   {final_val_acc*100:.2f}%")
    print(f"   Gap:            {final_gap*100:.2f}%")
    print(f"\n💾 Best model saved to: {save_path}")
    
    if final_gap > 0.1:
        print("\n⚠️  STILL OVERFITTING")
        print("   Next steps:")
        print("   1. Run augment_finnish.py again with target_count=100")
        print("   2. Increase Dropout to 0.6 in this script")
        print("   3. Collect more diverse test images")
    else:
        print("\n✅ Model is generalizing well!")
        print("   Test it on real Finnish products now!")

if __name__ == "__main__":
    main()