# train_model4_fixed.py — УЛЬТРА-МОДЕЛЬ (EfficientNetV2-L + TTA + CPU)
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === GPU ===
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
print("GPU:", tf.config.list_physical_devices('GPU'))

# === ПУТИ ===
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# === НАСТРОЙКИ ===
IMG_SIZE = 384
BATCH_SIZE = 16

# === АУГМЕНТАЦИЯ ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30.0,
    fill_mode='reflect'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# === ГЕНЕРАТОРЫ ===
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === Приведение типов к float32 (во избежание ошибки dtype='string') ===
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

# === МОДЕЛЬ ===
base = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(1024, activation='swish')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='swish')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(base.input, output)

# === ЭТАП 1: БЕЗ class_weight ===
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
print("\n=== ЭТАП 1: Замороженный EfficientNetV2-L ===")
model.fit(train_data, validation_data=val_data, epochs=10, verbose=1)

# === ЭТАП 2: Fine-tuning ===
base.trainable = True
for layer in base.layers[:200]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
]
print("\n=== ЭТАП 2: Fine-tuning ===")
model.fit(train_data, validation_data=val_data, epochs=30, callbacks=callbacks, verbose=1)

# === Функция TTA (Test-Time Augmentation) ===
def tta_predict(model, dataset, steps=8):
    preds = []
    for _ in range(steps):
        pred = model.predict(dataset, verbose=0)
        preds.append(pred)
    return np.mean(preds, axis=0)

print("\nTTA: Предсказание...")
tta_pred = tta_predict(model, test_data)
true_labels = np.concatenate([y for _, y in test_data], axis=0)
tta_acc = np.mean(np.argmax(tta_pred, axis=1) == np.argmax(true_labels, axis=1))
print(f"TTA ТОЧНОСТЬ: {tta_acc*100:.2f}%")

# === СОХРАНЕНИЕ ===
model.save('emotion_model5_v2l_ultra_fixed.h5')
print("\n✅ УЛЬТРА-МОДЕЛЬ СОХРАНЕНА: emotion_model5_v2l_ultra_fixed.h5")
