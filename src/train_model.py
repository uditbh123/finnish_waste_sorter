import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# 2. HYPERPARAMETERS
# ============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 20      # Train custom head
EPOCHS_FINE = 30      # Fine-tune top layers
LEARNING_RATE = 0.0001

# ============================================================================
# 3. DATA AUGMENTATION PIPELINE (SEPARATE FROM MODEL)
# ============================================================================
def get_augmentation_layer():
    """
    Returns augmentation pipeline for TRAINING data only.
    Applied to dataset, NOT baked into model architecture.
    """
    return tf.keras.Sequential([
        RandomFlip("horizontal"),           # Flip left/right
        RandomRotation(0.2),                # ±20% rotation
        RandomZoom(0.2),                    # ±20% zoom
        RandomContrast(0.3),                # Lighting variation
        tf.keras.layers.RandomBrightness(0.3),  # Brightness variation (fights color bias)
    ], name="data_augmentation")

# ============================================================================
# 4. MODEL ARCHITECTURE (CLEAN - NO AUGMENTATION)
# ============================================================================
def build_model(num_classes):
    """
    Builds a clean MobileNetV2 model for transfer learning.
    
    Architecture:
    - Input: 224x224x3
    - Base: MobileNetV2 (pretrained on ImageNet)
    - Head: GlobalAvgPool -> Dropout(0.5) -> Dense(256) -> Dropout(0.4) -> Dense(num_classes)
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        base_model: MobileNetV2 base (for unfreezing later)
        model: Complete model ready for training
    """
    # 1. Input Layer
    inputs = Input(shape=(224, 224, 3), name="input_image")

    # 2. Base Model (MobileNetV2 - pretrained on ImageNet)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    base_model.trainable = False  # Freeze initially

    # 3. Custom Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_pool")(x)
    
    # Strong regularization to prevent overfitting
    x = Dropout(0.5, name="dropout_1")(x)
    
    # Intermediate dense layer with L2 regularization
    x = Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01),
        name="dense_intermediate"
    )(x)
    
    x = Dropout(0.4, name="dropout_2")(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name="predictions")(x)

    # 4. Build complete model
    model = Model(inputs, outputs, name="waste_classifier")
    
    return base_model, model

# ============================================================================
# 5. TRAINING CALLBACKS
# ============================================================================
def get_callbacks(model_save_path):
    """
    Returns smart callbacks for training:
    - EarlyStopping: Stop if validation stops improving
    - ReduceLROnPlateau: Lower learning rate when stuck
    - ModelCheckpoint: Save best model automatically
    """
    return [
        # Stop training if validation loss doesn't improve for 8 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=False,  # ModelCheckpoint handles this
            verbose=1,
            mode='min'
        ),
        
        # Reduce learning rate if validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,           # Reduce LR by half
            patience=4,            # Wait 4 epochs before reducing
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        
        # Save best model based on validation loss
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,  # Save entire model
            verbose=1,
            mode='min'
        )
    ]

# ============================================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("🗑️  FINNISH WASTE CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n🚀 GPU Available: {len(gpus) > 0}")
    if len(gpus) > 0:
        print(f"   Using: {gpus[0].name}")
    
    # ========================================================================
    # STEP 1: LOAD DATASETS
    # ========================================================================
    print(f"\n📂 Loading dataset from: {DATA_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'  # Integer labels (0, 1, 2, ...)
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

    # Get class names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Found {num_classes} classes: {class_names}")

    # ========================================================================
    # STEP 2: APPLY AUGMENTATION & PREPROCESSING
    # ========================================================================
    print("\n🎨 Applying data augmentation to training set...")
    
    # Augmentation (training only)
    augmentation = get_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Normalization (both train and val)
    # MobileNetV2 expects pixel values in [-1, 1] range
    normalization = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    
    print("📊 Normalizing datasets (scaling to [-1, 1])...")
    train_ds = train_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_ds = val_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Performance optimization
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("✅ Data pipeline ready!")

    # ========================================================================
    # STEP 3: BUILD MODEL
    # ========================================================================
    print("\n🏗️  Building model architecture...")
    base_model, model = build_model(num_classes)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Show architecture
    print("\n📊 Model Summary:")
    model.summary()
    
    print(f"\n📈 Total parameters: {model.count_params():,}")
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"   Trainable: {trainable:,}")
    print(f"   Frozen: {model.count_params() - trainable:,}")

    # ========================================================================
    # STEP 4: PHASE 1 - TRAIN CUSTOM HEAD
    # ========================================================================
    save_path = os.path.join(MODEL_DIR, "waste_sorter_best.keras")
    
    print("\n" + "=" * 70)
    print(f"🧠 PHASE 1: Training Custom Head ({EPOCHS_HEAD} epochs)")
    print("=" * 70)
    print("   Strategy: Keep MobileNetV2 frozen, train only custom layers")
    print("   Learning Rate:", LEARNING_RATE)
    print("   Early stopping: patience=8 epochs")
    print()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(save_path),
        verbose=1
    )

    # Report Phase 1 results
    print("\n" + "=" * 70)
    print("📊 PHASE 1 RESULTS")
    print("=" * 70)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc
    
    print(f"   Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"   Val Accuracy:   {final_val_acc*100:.2f}%")
    print(f"   Gap:            {gap*100:.2f}%")
    
    if gap > 0.15:
        print("   ⚠️  High overfitting detected (gap > 15%)")
    elif gap > 0.10:
        print("   ⚠️  Moderate overfitting (gap 10-15%)")
    else:
        print("   ✅ Good generalization!")

    # ========================================================================
    # STEP 5: PHASE 2 - FINE-TUNING
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"🔓 PHASE 2: Fine-Tuning Top Layers ({EPOCHS_FINE} epochs)")
    print("=" * 70)
    
    # Unfreeze base model
    base_model.trainable = True

    # Freeze bottom 100 layers (keep low-level features stable)
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_layers = sum([1 for layer in model.layers if layer.trainable])
    print(f"   Unfreezing top {len(base_model.layers) - fine_tune_at} layers of MobileNetV2")
    print(f"   Total trainable layers: {trainable_layers}")
    print(f"   Learning Rate: {LEARNING_RATE / 10} (10x lower)")
    print()

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tune
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks(save_path),
        verbose=1
    )

    # ========================================================================
    # STEP 6: FINAL REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    final_train_acc = history_fine.history['accuracy'][-1]
    final_val_acc = history_fine.history['val_accuracy'][-1]
    final_gap = final_train_acc - final_val_acc
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"   Val Accuracy:   {final_val_acc*100:.2f}%")
    print(f"   Overfitting Gap: {final_gap*100:.2f}%")
    
    print(f"\n💾 Best model saved to:")
    print(f"   {save_path}")
    print(f"\n📝 To load model:")
    print(f"   model = tf.keras.models.load_model('{save_path}')")
    
    # Give recommendations
    print("\n" + "=" * 70)
    if final_gap > 0.15:
        print("⚠️  STILL OVERFITTING SIGNIFICANTLY")
        print("\n   Recommended next steps:")
        print("   1. Collect more diverse training data (especially Finnish products)")
        print("   2. Increase Dropout to 0.6 in build_model()")
        print("   3. Add more aggressive augmentation")
        print("   4. Consider using MobileNetV3 or EfficientNet")
    elif final_gap > 0.10:
        print("⚠️  MODERATE OVERFITTING")
        print("\n   Recommended next steps:")
        print("   1. Test on real Finnish products")
        print("   2. Collect edge cases that fail")
        print("   3. Fine-tune augmentation strategy")
    else:
        print("✅ MODEL IS GENERALIZING WELL!")
        print("\n   Next steps:")
        print("   1. Test on real Finnish products (Valio, Atria, etc.)")
        print("   2. Build inference script for real-time classification")
        print("   3. Deploy to mobile app or web interface")
    
    print("=" * 70)
    print("\n🚀 Ready for testing! Run: python src/test_model.py")

# ============================================================================
# 7. ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()