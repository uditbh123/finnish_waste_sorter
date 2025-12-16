import os
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
EPOCHS = 8 # How many times the AI Studies the data (takes 10-15mins)

# 3. Data Augmentation (The "Magic")
#This creates fake variations (zoomed, rotated) to make the AI smarter.
# It is critical for your small "Mixed" waste category.
train_datagen = ImageDataGenerator(
    rescale=1./255, #Normalize pixel values (0-1 instead of 0-225)
    rotation_range = 20, #Rotate slightly
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

# Print the class mapping to know which ID is which (e.g., 0=Biowaste, 1=Glass)
print(f"✅ Classes found: {train_generator.class_indices}")

# 5. Build the Model (transfer Learning)
print("🏗️ Building the Model...")

# Load MobileNetv2 (pre-trained on ImageNet) but Cut Off the top (head)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base (so we don't destroy the pre-trained knowledge)
base_model.trainable = False 

# Add our own "Finnish Recycling Head"
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) # Prevents overfitting (memorizing instead of learning)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile 
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. train
print("🚀 Starting training... (This will take time!)")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 8. Save the Brain
print("💾 Saving the model...")
model_path = os.path.join(MODEL_DIR, "waste_sorter_model.h5")
model.save(model_path)
print(f"🎉 Model saved successfully at: {model_path}")