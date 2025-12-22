import os
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 

# 1. Setup paths 
# This finds my project folder automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create the models folder if it doesn't exist 
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 8 # fast learning 
EPOCHS_PHASE_2 = 8 # slow, careful fine-tuning

# 3. Data Augmentation (The "Magic")
#This creates fake variations (zoomed, rotated) to make the AI smarter.
# It is critical for your small "Mixed" waste category.
train_datagen = ImageDataGenerator(
    rescale=1./255, #Normalize pixel values (0-1 instead of 0-225)
    rotation_range = 30, #Rotate slightly
    width_shift_range = 0.2, # Move left/right
    height_shift_range = 0.2, # Move up/down
    horizontal_flip = True, # Mirror Image 
    fill_mode='nearest', 
    validation_split = 0.2 # Use 20% of data for testing
)

print("⏳ Loading images... this might take a moment.")

# 4. Load data (Train & Validation Split)
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

# Build Model (Phase 1: Frozen Base)
print("\n🏗️ Building Model (Phase 1)...")
base_model = MobileNetV2(weights='imagenet', include_top =False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze the base 

x = base_model.output 
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            
# 5. Train Phase 1
print("🚀 Starting Phase 1 Training (Standard)...")
history = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_1,
    validation_data=validation_generator
)

# 6. Fine-Tuning (phase2: Unfrozen)
print("\n🔓 Unfreezing top layers for Fine-Tuning...")
base_model.trainable = True

# Freezing all layers except the last 30
# This allows the AI to adjust its "eyes" to see finnish shapes
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with VERY LOW learning rate (so we don't break what it already knows)
model.compile(optimizer=Adam(learning_rate=1e-5),  # 100x slower learning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Starting Phase 2 Training (Fine-Tuning)...")
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE_2,
    validation_data=validation_generator
)

# 7. Save
print("💾 Saving the Smart Model...")
model_path = os.path.join(MODEL_DIR, "waste_sorter_model.h5")
model.save(model_path)
print(f"🎉 Model saved successfully at: {model_path}")