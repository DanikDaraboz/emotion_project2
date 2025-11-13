import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# === ФИКС ДЛЯ GPU НА WINDOWS (WDDM) ===
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("=== TensorFlow Info ===")
print("Версия TensorFlow:", tf.__version__)
print("Построен с CUDA:", tf.test.is_built_with_cuda())

# Принудительная инициализация GPU
print("Инициализация GPU...")
try:
    with tf.device('/GPU:0'):
        _ = tf.constant([[1.0]])
    print("GPU УСПЕШНО ИНИЦИАЛИЗИРОВАН!")
except Exception as e:
    print("Не удалось использовать GPU:", e)

print("GPU устройства:", tf.config.list_physical_devices('GPU'))

# === Настройки ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 15   # Первый этап
EPOCHS_FINE = 30      # Fine-tuning

# === Пути ===
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# === УЛУЧШЕННАЯ АУГМЕНТАЦИЯ ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# === Генераторы ===
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === ПРОВЕРКА БАЛАНСА КЛАССОВ ===
class_labels = list(train_data.class_indices.keys())
class_counts = np.bincount(train_data.classes)
print("Распределение классов:", dict(zip(class_labels, class_counts)))

# Веса классов
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Веса классов:", class_weights)

# === ЭТАП 1: Обучение с замороженным MobileNetV2 ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("ЭТАП 1: Обучение с замороженным MobileNetV2...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_INITIAL,
    class_weight=class_weights,
    verbose=1
)

# === ЭТАП 2: Fine-tuning (разморозка верхних слоёв) ===
print("ЭТАП 2: Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# === ТЕСТ ===
print("Оценка на тестовых данных...")
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"ТОЧНОСТЬ НА ТЕСТЕ: {test_acc:.4f} ({test_acc*100:.2f}%)")

# === СОХРАНЕНИЕ В .H5 ФОРМАТЕ (как раньше) ===
model.save('emotion_model2_mobilenetv2_finetuned.h5')
print("МОДЕЛЬ СОХРАНЕНА: emotion_model_mobilenetv2_finetuned.h5")