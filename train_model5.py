# train_model5_fast.py — Быстрая и точная модель (MobileNetV3 + LabelSmoothing)
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === GPU / CPU ===
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
print("GPU:", tf.config.list_physical_devices('GPU'))

# === ПУТИ ===
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# === НАСТРОЙКИ ===
IMG_SIZE = 224
BATCH_SIZE = 32

# === АУГМЕНТАЦИЯ ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)

# === Dataset wrapper для TensorFlow ===
train_data = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )
)
val_data = tf.data.Dataset.from_generator(
    lambda: val_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )
)
test_data = tf.data.Dataset.from_generator(
    lambda: test_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )
)

# === МОДЕЛЬ (MobileNetV3-Large) ===
base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(512, activation='swish')(x)
x = Dropout(0.4)(x)
output = Dense(4, activation='softmax')(x)
model = Model(base.input, output)

# === КОМПИЛЯЦИЯ ===
model.compile(
    optimizer=Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

print("\n=== ЭТАП 1: Замороженная MobileNetV3 ===")
model.fit(train_data, validation_data=val_data, epochs=10, verbose=1)

# === ЭТАП 2: Fine-tuning последних слоёв ===
base.trainable = True
for layer in base.layers[:-60]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
]

print("\n=== ЭТАП 2: Fine-tuning ===")
model.fit(train_data, validation_data=val_data, epochs=20, callbacks=callbacks, verbose=1)

# === ОЦЕНКА ===
print("\nОценка точности:")
acc = model.evaluate(test_data, verbose=1)[1] * 100
print(f"✅ Финальная точность: {acc:.2f}%")

# === СОХРАНЕНИЕ ===
model.save('emotion_model_fast.h5')
print("\n🚀 МОДЕЛЬ СОХРАНЕНА: emotion_model_fast.h5")
