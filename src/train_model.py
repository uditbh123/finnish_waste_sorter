import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from sklearn.utils.class_weight import compute_class_weight

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 8
EPOCHS_PHASE_2 = 8

# 3. Data Loading 
# I use standard generator because I have already created augmented files in Step 1
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("⏳ Loading images...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 4. Compute Class Weights (To fix imbalance)
print("⚖️  Computing Class Weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"   Weights: {class_weight_dict}")

# 5. Build Model
print("\n🏗️  Building Model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile with FOCAL LOSS
# This forces the model to focus on the hard "Valio" cases
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
              metrics=['accuracy'])

# 7. Train Phase 1
print("🚀 Phase 1: Training Head...")
history = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_1,
    validation_data=validation_generator
    #class_weight=class_weight_dict 
)

# 8. Train Phase 2 (Fine-Tuning)
print("\n🔓 Phase 2: Unfreezing top layers...")
base_model.trainable = True
for layer in base_model.layers[:-40]: # Only train top 40 layers
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_2,
    validation_data=validation_generator
    #class_weight=class_weight_dict
)

# 9. Save
model.save(os.path.join(MODEL_DIR, "waste_sorter_model.h5"))
print("🎉 Advanced Model Saved.")