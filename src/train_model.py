import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation
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
MIXUP_ALPHA = 0.4

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
        RandomTranslation(0.15, 0.15),      # Object moves around frame (pile-like scenes)
        RandomContrast(0.3),                # Lighting variation
        tf.keras.layers.RandomBrightness(0.3),  # Brightness variation (fights color bias)
    ], name="data_augmentation")


def mixup_batch(images, labels, alpha=MIXUP_ALPHA):
    """Batch-wise MixUp to reduce background overfitting and pile confusion."""
    batch_size = tf.shape(images)[0]
    shuffle_idx = tf.random.shuffle(tf.range(batch_size))

    mixed_images_2 = tf.gather(images, shuffle_idx)
    mixed_labels_2 = tf.gather(labels, shuffle_idx)

    # Beta(alpha, alpha) sampled using Gamma trick
    gamma_1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma_2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lam = gamma_1 / (gamma_1 + gamma_2)

    lam_img = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_lbl = tf.reshape(lam, [batch_size, 1])

    mixed_images = images * lam_img + mixed_images_2 * (1.0 - lam_img)
    mixed_labels = labels * lam_lbl + mixed_labels_2 * (1.0 - lam_lbl)
    return mixed_images, mixed_labels

# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================
def build_model(num_classes):
    """
    Builds a clean MobileNetV2 model for transfer learning.
    """
    # 1. Input Layer
    inputs = Input(shape=(224, 224, 3), name="input_image")

    # 2. Base Model (MobileNetV2 - pretrained on ImageNet)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    base_model.trainable = False

    # 3. Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax', name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="waste_classifier")
    return base_model, model

def get_callbacks(save_path):
    return [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

# ============================================================================
# 5. ENTRY POINT
# ============================================================================
def main():
    print("🚀 Starting SortWise Training Pipeline...")

    # Load datasets with one-hot encoding for Focal Loss & Mixup
    print("\n📂 Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
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

    # MixUp helps with multi-object / pile scenes
    train_ds = train_ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)

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
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
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
        callbacks=get_callbacks(save_path)
    )

    # ========================================================================
    # STEP 5: PHASE 2 - FINE-TUNING
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"🔓 PHASE 2: Fine-Tuning Top Layers ({EPOCHS_FINE} epochs)")
    print("=" * 70)
    
    # Unfreeze base model
    base_model.trainable = True

    # Freeze bottom 100 layers
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
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
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

    if final_gap > 0.20:
        print("⚠️  SEVERE OVERFITTING DETECTED")
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
    print("\n🚀 Ready for testing! Run: python src/predict.py")

if __name__ == "__main__":
    main()