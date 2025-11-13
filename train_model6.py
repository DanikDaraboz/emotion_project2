# train_model_v2_fastaccurate.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==========================
# 🔧 НАСТРОЙКА ОКРУЖЕНИЯ
# ==========================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("\n=== TensorFlow Info ===")
print("Версия TensorFlow:", tf.__version__)
print("CUDA доступен:", tf.test.is_built_with_cuda())
print("Доступные GPU:", tf.config.list_physical_devices('GPU'))

# ==========================
# 📁 ПУТИ К ДАННЫМ
# ==========================
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# ==========================
# ⚙️ НАСТРОЙКИ
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 64    # быстрее на CPU/GPU при большом датасете
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15

# ==========================
# 📈 АУГМЕНТАЦИЯ
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ==========================
# 🧠 СОЗДАНИЕ МОДЕЛИ
# ==========================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # сначала заморозим

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='swish')(x)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# ==========================
# 📦 КОМПИЛЯЦИЯ (Этап 1)
# ==========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_stage1 = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint("best_model_v2_stage1.h5", save_best_only=True)
]

print("\n=== ЭТАП 1: Базовое обучение (замороженные слои) ===")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_STAGE1, callbacks=callbacks_stage1, verbose=1)

# ==========================
# 🔧 FINE-TUNING (Этап 2)
# ==========================
print("\n=== ЭТАП 2: Тонкая настройка последних слоёв ===")
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

callbacks_stage2 = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint("best_model_v2_finetuned.h5", save_best_only=True)
]

model.fit(train_data, validation_data=val_data, epochs=EPOCHS_STAGE2, callbacks=callbacks_stage2, verbose=1)

# ==========================
# 🧩 ОЦЕНКА
# ==========================
print("\nОЦЕНКА ТОЧНОСТИ:")
loss, acc = model.evaluate(test_data)
print(f"✅ Финальная точность: {acc*100:.2f}%")

# ==========================
# 💾 СОХРАНЕНИЕ
# ==========================
model.save("emotion_model5_v2_fastaccurate.h5")
print("\n🚀 Модель сохранена: emotion_model5_v2_fastaccurate.h5")
