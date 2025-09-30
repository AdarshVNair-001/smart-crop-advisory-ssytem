# =========================
# 1️⃣ IMPORT LIBRARIES
# =========================
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# =========================
# 2️⃣ PARAMETERS
# =========================
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20

# Parent folder containing both datasets
DATASET_DIR = r'D:\Users\Adarsh V Nair\Documents\projects\final yr\smart-crop-advisory-system\dataset'

# Model save path
MODEL_DIR = os.path.join(os.path.dirname(DATASET_DIR), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_disease_pest_model.h5')

# =========================
# 3️⃣ DATA AUGMENTATION & LOADING
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% validation
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# =========================
# 4️⃣ BUILD CNN MODEL (TRANSFER LEARNING)
# =========================
num_classes = len(train_gen.class_indices)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model for initial training
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =========================
# 5️⃣ TRAIN THE MODEL
# =========================
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# =========================
# 6️⃣ SAVE MODEL
# =========================
model.save(MODEL_PATH)
print(f'Model saved at: {MODEL_PATH}')

# =========================
# 7️⃣ RECOMMENDATION SYSTEM
# =========================
# Map all disease + pest classes to actionable advice
recommendations = {
    'Tomato___Early_blight': 'Apply fungicide X, remove infected leaves',
    'Tomato___Late_blight': 'Apply fungicide Y, improve air circulation',
    'Caterpillar': 'Use Bacillus thuringiensis or neem oil',
    'Aphid': 'Use insecticidal soap or neem oil',
    'Healthy': 'No action needed'
}

# =========================
# 8️⃣ PREDICTION ON NEW IMAGE
# =========================
def predict_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class = list(train_gen.class_indices.keys())[np.argmax(pred)]
    action = recommendations.get(pred_class, 'No recommendation')

    return pred_class, action

# Example
pred_class, action = predict_image(r'D:\Users\Adarsh V Nair\Documents\projects\final yr\smart-crop-advisory-system\test_images\test_image.jpg')
print(f'Predicted class: {pred_class}')
print(f'Recommended action: {action}')

