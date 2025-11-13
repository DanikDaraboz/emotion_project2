import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# === ФИКС GPU (WDDM) ===
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("=== TensorFlow Info ===")
print("Версия TF:", tf.__version__)

# === Настройки ===
IMG_SIZE = 300  # EfficientNetB3 любит 300x300
BATCH_SIZE = 16  # Меньше — из-за памяти (12M параметров)
EPOCHS_INITIAL = 10
EPOCHS_FINE = 25

# === Пути ===
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# === АУГМЕНТАЦИЯ ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# === Генераторы ===
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)

# === ВЕСА КЛАССОВ ===
class_weights = compute_class_weight('balanced', classes=np.unique(train_data.classes), y=train_data.classes)
class_weights = dict(enumerate(class_weights))

# === ЭТАП 1: Замороженный EfficientNetB3 ===
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("ЭТАП 1: Обучение с замороженным EfficientNetB3...")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_INITIAL, class_weight=class_weights)

# === ЭТАП 2: Fine-tuning ===
base_model.trainable = True
for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
]

print("ЭТАП 2: Fine-tuning...")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FINE, class_weight=class_weights, callbacks=callbacks)

# === ТЕСТ ===
test_loss, test_acc = model.evaluate(test_data)
print(f"ТОЧНОСТЬ: {test_acc*100:.2f}%")

# === СОХРАНЕНИЕ В .H5 ===
model.save('emotion_model3_efficientnetb3.h5')
print("МОДЕЛЬ СОХРАНЕНА: emotion_model_efficientnetb3.h5")