import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from scipy import ndimage
import os


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'  # Разрешает GPU в WDDM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'       # Убирает лишние логи

print("=== TensorFlow Info ===")
print("Версия TensorFlow:", tf.__version__)
print("Построен с CUDA:", tf.test.is_built_with_cuda())

# Принудительно "разбудим" GPU
print("Инициализация GPU...")
try:
    with tf.device('/GPU:0'):
        _ = tf.constant([[1.0]])
    print("GPU УСПЕШНО ИНИЦИАЛИЗИРОВАН!")
except Exception as e:
    print("Не удалось использовать GPU:", e)

# Проверяем, видит ли TF GPU после инициализации
print("GPU устройства после инициализации:", tf.config.list_physical_devices('GPU'))

# === Настройки ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# === Пути к папкам ===
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# === Генераторы данных ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

# === Загружаем базовую модель MobileNetV2 ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # замораживаем веса

# === Добавляем классификатор ===
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)  # 4 эмоции

model = Model(inputs=base_model.input, outputs=output)

# === Компиляция ===
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Обучение ===
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# === Тест ===
test_loss, test_acc = model.evaluate(test_data)
print(f"🎯 Точность на тестовых данных: {test_acc:.2f}")

# === Сохраняем модель ===
model.save('emotion_model3_mobilenetv2.h5')
print("✅ Модель сохранена как emotion_model3_mobilenetv2.h5")